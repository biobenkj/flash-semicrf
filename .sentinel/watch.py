"""Watch/daemon infrastructure — file monitoring, debouncing, event emission."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from core import CodeSentinel
from fix import _fix_traces
from models import Status, TraceVerification, WatchEvent
from output import _now_iso, _write_status_json


class _FileDebouncer:
    """Debounce rapid filesystem events per-file."""

    def __init__(self, delay_ms: int = 500):
        self.delay_s = delay_ms / 1000.0
        self._pending: dict[str, float] = {}

    def record(self, path: str) -> None:
        """Record a file change event. Resets the timer if already pending."""
        self._pending[path] = time.monotonic()

    def ready(self) -> list[str]:
        """Return paths whose debounce delay has elapsed, removing them from pending."""
        now = time.monotonic()
        fired = [p for p, t in self._pending.items() if now - t >= self.delay_s]
        for p in fired:
            del self._pending[p]
        return fired

    def has_pending(self) -> bool:
        return bool(self._pending)


def _emit_watch_event(event: WatchEvent, json_lines: bool) -> None:
    """Emit a watch event. JSON Lines to stdout if json_lines, else human-readable to stderr."""
    if json_lines:
        # Strip None fields for cleaner output
        data = {k: v for k, v in asdict(event).items() if v is not None}
        print(json.dumps(data), flush=True)
    else:
        ts = event.timestamp
        if event.event == "status_snapshot":
            print(f"[{ts}] sentinel watch: {event.summary}", file=sys.stderr)
        elif event.event == "drift_detected":
            fixable = " (auto-fixable)" if event.auto_fixable else " (manual review)"
            print(
                f"[{ts}] drift: {event.trace} [{', '.join(event.anchors_affected or [])}]{fixable}",
                file=sys.stderr,
            )
        elif event.event == "fix_applied":
            print(
                f"[{ts}] fixed: {event.trace} [{', '.join(event.anchors_updated or [])}] -> {event.new_status}",
                file=sys.stderr,
            )
        elif event.event == "semantic_drift":
            print(
                f"[{ts}] SEMANTIC DRIFT: {event.trace} [{', '.join(event.anchors_affected or [])}] — {event.message}",
                file=sys.stderr,
            )
        elif event.event == "trace_verified":
            print(f"[{ts}] verified: {event.trace}", file=sys.stderr)
        elif event.event == "shutdown":
            print(f"[{ts}] shutdown: {event.summary}", file=sys.stderr)


# --- Git event detection for watch ---

_GIT_WATCH_FILES = {"HEAD", "index"}
_GIT_WATCH_DIRS = {"refs/heads"}


def _is_relevant_git_event(path: str, repo_root: Path) -> bool:
    """Check if a git path change should trigger re-verification.

    Relevant: .git/HEAD (branch switch), .git/refs/heads/* (new commits),
              .git/index (staging).
    Irrelevant: .git/objects/* (constant writes), .git/logs/*, etc.
    """
    git_dir = str((repo_root / ".git").resolve())
    if not path.startswith(git_dir):
        return False
    rel = path[len(git_dir) :].lstrip(os.sep)
    if rel in _GIT_WATCH_FILES:
        return True
    return any(rel.startswith(d) for d in _GIT_WATCH_DIRS)


def _git_watch_paths(repo_root: Path) -> list[str]:
    """Specific git paths worth polling for changes."""
    git = repo_root / ".git"
    paths = [str(git / "HEAD"), str(git / "index")]
    refs_dir = git / "refs" / "heads"
    if refs_dir.is_dir():
        for ref in refs_dir.iterdir():
            if ref.is_file():
                paths.append(str(ref))
    return paths


def _handle_file_change(
    sentinel: CodeSentinel,
    changed_files: list[str],
    file_trace_index: dict[str, list[str]],
    auto_fix: bool,
    json_lines: bool,
    verification_cache: dict[str, TraceVerification],
) -> None:
    """Process file changes: re-verify affected traces, auto-fix if possible."""
    # Deduplicate affected traces
    affected_traces: set[str] = set()
    for path in changed_files:
        affected_traces.update(file_trace_index.get(path, []))

    if not affected_traces:
        return

    for trace_name in affected_traces:
        verification = sentinel.verify_trace(trace_name)
        verification_cache[trace_name] = verification

        if verification.overall_status == Status.VERIFIED:
            _emit_watch_event(
                WatchEvent(event="trace_verified", timestamp=_now_iso(), trace=trace_name),
                json_lines,
            )
            continue

        # Drift detected — classify anchors
        broken_anchors = [a.name for a in verification.anchors if a.status != "VERIFIED"]
        auto_fixable = all(
            a.content_hash_match is not False
            for a in verification.anchors
            if a.status != "VERIFIED"
        )

        _emit_watch_event(
            WatchEvent(
                event="drift_detected",
                timestamp=_now_iso(),
                trace=trace_name,
                anchors_affected=broken_anchors,
                auto_fixable=auto_fixable,
            ),
            json_lines,
        )

        if auto_fix and auto_fixable:
            _exit, _modified, final_verifications, fix_results = _fix_traces(sentinel, [trace_name])
            result = fix_results.get(trace_name, {})
            fixed_names = [af["name"] for af in result.get("anchors_fixed", [])]
            new_status = result.get("new_status", "DEGRADED")

            # Update cache with post-fix verification
            for fv in final_verifications:
                if fv.name == trace_name:
                    verification_cache[trace_name] = fv
                    break

            _emit_watch_event(
                WatchEvent(
                    event="fix_applied",
                    timestamp=_now_iso(),
                    trace=trace_name,
                    anchors_updated=fixed_names,
                    new_status=new_status,
                ),
                json_lines,
            )
        elif broken_anchors:
            failed_names = [
                a.name
                for a in verification.anchors
                if a.status != "VERIFIED" and a.content_hash_match is False
            ]
            _emit_watch_event(
                WatchEvent(
                    event="semantic_drift",
                    timestamp=_now_iso(),
                    trace=trace_name,
                    anchors_affected=failed_names or broken_anchors,
                    message="Content hash mismatch — manual review required",
                ),
                json_lines,
            )

    # Write status.json from full cache
    _write_status_json(sentinel, list(verification_cache.values()))


def _handle_git_event(
    sentinel: CodeSentinel,
    json_lines: bool,
    verification_cache: dict[str, TraceVerification],
    rebuild_index_fn,
) -> None:
    """Handle git events (commit, checkout, rebase): rebuild index and re-verify all."""
    rebuild_index_fn()

    # Re-verify all traces
    meta = sentinel.load_meta()
    verification_cache.clear()
    for trace_name in meta.get("sentinels", {}):
        verification_cache[trace_name] = sentinel.verify_trace(trace_name)

    _write_status_json(sentinel, list(verification_cache.values()))

    verified = sum(1 for v in verification_cache.values() if v.overall_status == Status.VERIFIED)
    total = len(verification_cache)
    _emit_watch_event(
        WatchEvent(
            event="status_snapshot",
            timestamp=_now_iso(),
            traces={n: v.overall_status.value for n, v in verification_cache.items()},
            summary=f"{verified}/{total} verified",
        ),
        json_lines,
    )


def _watch_with_watchdog(
    sentinel: CodeSentinel,
    file_trace_index: dict[str, list[str]],
    debouncer: _FileDebouncer,
    shutdown_flag,
    auto_fix: bool,
    json_lines: bool,
    verification_cache: dict[str, TraceVerification],
) -> None:
    """Watch loop using watchdog (preferred backend)."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    # Mutable set so the handler thread always sees the current state.
    # Rebinding via nonlocal would leave the handler holding a stale reference.
    monitored_paths = set(file_trace_index.keys())

    class SentinelHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.is_directory:
                return
            path = str(Path(event.src_path).resolve())
            if path in monitored_paths:
                debouncer.record(path)
            elif _is_relevant_git_event(path, sentinel.repo_root):
                debouncer.record(path)

    observer = Observer()
    handler = SentinelHandler()

    # Watch only directories containing monitored files (not entire repo)
    for d in sentinel.get_monitored_dirs():
        if os.path.isdir(d):
            observer.schedule(handler, d, recursive=False)
    # Watch .git/ for commit/checkout events
    git_dir = str(sentinel.repo_root / ".git")
    if os.path.isdir(git_dir):
        observer.schedule(handler, git_dir, recursive=False)
    git_refs_dir = str(sentinel.repo_root / ".git" / "refs" / "heads")
    if os.path.isdir(git_refs_dir):
        observer.schedule(handler, git_refs_dir, recursive=True)

    observer.start()
    try:
        while not shutdown_flag():
            ready = debouncer.ready()
            if ready:
                git_changed = any(_is_relevant_git_event(p, sentinel.repo_root) for p in ready)
                source_changed = [p for p in ready if p in monitored_paths]
                if git_changed:

                    def rebuild():
                        nonlocal file_trace_index
                        file_trace_index = sentinel.build_file_trace_index()
                        # Mutate in-place — handler thread holds a reference to this set
                        monitored_paths.clear()
                        monitored_paths.update(file_trace_index.keys())
                        # TODO: reschedule observer on index rebuild — a branch switch
                        # could add source files in directories not currently watched.
                        # For now, those files won't be detected until next restart.
                        return file_trace_index

                    _handle_git_event(sentinel, json_lines, verification_cache, rebuild)
                elif source_changed:
                    _handle_file_change(
                        sentinel,
                        source_changed,
                        file_trace_index,
                        auto_fix,
                        json_lines,
                        verification_cache,
                    )
            time.sleep(0.1)
    finally:
        observer.stop()
        observer.join()


def _watch_with_polling(
    sentinel: CodeSentinel,
    file_trace_index: dict[str, list[str]],
    debouncer: _FileDebouncer,
    shutdown_flag,
    auto_fix: bool,
    json_lines: bool,
    verification_cache: dict[str, TraceVerification],
    poll_interval: float,
) -> None:
    """Watch loop using stat-based polling (fallback when watchdog unavailable)."""
    mtimes: dict[str, float] = {}

    # Initialize mtimes for monitored files + git paths
    all_paths = list(file_trace_index.keys()) + _git_watch_paths(sentinel.repo_root)
    for path in all_paths:
        try:
            mtimes[path] = os.stat(path).st_mtime
        except OSError:
            pass

    monitored_paths = set(file_trace_index.keys())

    while not shutdown_flag():
        for path, old_mtime in list(mtimes.items()):
            try:
                new_mtime = os.stat(path).st_mtime
                if new_mtime != old_mtime:
                    mtimes[path] = new_mtime
                    debouncer.record(path)
            except OSError:
                pass

        ready = debouncer.ready()
        if ready:
            git_changed = any(_is_relevant_git_event(p, sentinel.repo_root) for p in ready)
            source_changed = [p for p in ready if p in monitored_paths]
            if git_changed:

                def rebuild():
                    nonlocal file_trace_index
                    file_trace_index = sentinel.build_file_trace_index()
                    # Mutate in-place for consistency with watchdog backend
                    monitored_paths.clear()
                    monitored_paths.update(file_trace_index.keys())
                    # Rebuild mtimes for new file set
                    mtimes.clear()
                    new_all = list(file_trace_index.keys()) + _git_watch_paths(sentinel.repo_root)
                    for p in new_all:
                        try:
                            mtimes[p] = os.stat(p).st_mtime
                        except OSError:
                            pass
                    return file_trace_index

                _handle_git_event(sentinel, json_lines, verification_cache, rebuild)
            elif source_changed:
                _handle_file_change(
                    sentinel,
                    source_changed,
                    file_trace_index,
                    auto_fix,
                    json_lines,
                    verification_cache,
                )

        time.sleep(poll_interval)


def _check_stale_pidfile(sentinel: CodeSentinel) -> None:
    """Clean up stale PID file from previous crash, or exit if watch is already running."""
    pid_path = sentinel.sentinel_dir / "watch.pid"
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # Check if process exists (doesn't actually kill)
        print(f"Error: watch is already running (PID {pid})", file=sys.stderr)
        print(f"If this is stale, remove {pid_path}", file=sys.stderr)
        sys.exit(1)
    except (ProcessLookupError, ValueError):
        # Process doesn't exist — stale PID file
        pid_path.unlink(missing_ok=True)
    except PermissionError:
        # Process exists but we can't signal it — assume it's running
        print("Error: watch may be running (PID file exists, permission denied)", file=sys.stderr)
        sys.exit(1)


def _cleanup_pidfile(sentinel: CodeSentinel) -> None:
    """Remove PID file on shutdown."""
    pid_path = sentinel.sentinel_dir / "watch.pid"
    pid_path.unlink(missing_ok=True)


def _daemonize(sentinel: CodeSentinel) -> None:
    """Fork to background and write PID file. Unix only."""
    if sys.platform == "win32":
        print(
            "Error: --daemon is not supported on Windows. "
            "The polling backend works fine — only daemonization requires Unix.",
            file=sys.stderr,
        )
        sys.exit(1)

    pid = os.fork()
    if pid > 0:
        # Parent exits
        sys.exit(0)

    # Child continues as daemon
    os.setsid()
    pid_path = sentinel.sentinel_dir / "watch.pid"
    pid_path.write_text(str(os.getpid()))

    # Redirect stdout/stderr to log file
    log_path = sentinel.sentinel_dir / "watch.log"
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)
