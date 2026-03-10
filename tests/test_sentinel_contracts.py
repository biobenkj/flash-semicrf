import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest

_SENTINEL_PATH = Path(__file__).resolve().parents[1] / ".sentinel" / "sentinel.py"

pytestmark = [
    pytest.mark.diagnostic,
    pytest.mark.skipif(
        not _SENTINEL_PATH.exists(),
        reason="requires .sentinel/sentinel.py in repository checkout",
    ),
]


def _load_sentinel_module():
    path = _SENTINEL_PATH
    unique_name = f"sentinel_cli_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sentinel module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSentinel:
    def __init__(self, verify_result, config=None):
        self._verify_result = verify_result
        self.config = config or {}

    def load_meta(self):
        return {"sentinels": {"trace-a": {}}}

    def check_consistency(self):
        return []

    def verify_trace(self, trace_name, check_assumptions=True):  # noqa: ARG002
        return self._verify_result


class _AdapterSentinel:
    def __init__(self, repo_root: Path, sentinel_dir: Path, config: dict):
        self.repo_root = repo_root
        self.sentinel_dir = sentinel_dir
        self.config = config


def _make_trace_result(
    module,
    *,
    commit_status,
    anchors=None,
    assumptions=None,
    verified_commit="abc123",
    current_commit="def456",
):
    return module.TraceVerification(
        name="trace-a",
        commit_status=commit_status,
        verified_commit=verified_commit,
        current_commit=current_commit,
        uncommitted_changes=False,
        anchors=anchors or [],
        assumptions=assumptions or [],
    )


def test_verify_json_error_when_trace_selector_missing(capsys):
    module = _load_sentinel_module()
    sentinel = _FakeSentinel(
        _make_trace_result(module, commit_status=module.Status.VERIFIED),
        config={"ci": {"strict_mode": False}},
    )
    args = argparse.Namespace(
        all=False,
        trace=None,
        check_consistency=False,
        format="json",
        strict=False,
        no_strict=False,
    )

    exit_code = module.cmd_verify(args, sentinel)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == module.EXIT_GENERAL_ERROR
    assert payload["command"] == "verify"
    assert payload["exit_code"] == module.EXIT_GENERAL_ERROR
    assert payload["error"] == "Specify --trace NAME, --all, or --affected-by FILE"


def test_verify_missing_trace_is_nonzero_and_missing_state(capsys):
    module = _load_sentinel_module()
    sentinel = _FakeSentinel(
        _make_trace_result(
            module,
            commit_status=module.Status.MISSING,
            verified_commit="",
            current_commit=None,
        ),
        config={"ci": {"strict_mode": False}},
    )
    args = argparse.Namespace(
        all=False,
        trace="does-not-exist",
        check_consistency=False,
        format="json",
        strict=False,
        no_strict=False,
    )

    exit_code = module.cmd_verify(args, sentinel)
    payload = json.loads(capsys.readouterr().out)

    trace_payload = payload["traces"]["does-not-exist"]
    assert exit_code == module.EXIT_GENERAL_ERROR
    assert payload["exit_code"] == module.EXIT_GENERAL_ERROR
    assert trace_payload["status"] == "MISSING"
    assert trace_payload["commit_status"] == "MISSING"


def test_verify_conflicting_strict_flags_returns_json_error(capsys):
    module = _load_sentinel_module()
    sentinel = _FakeSentinel(
        _make_trace_result(module, commit_status=module.Status.VERIFIED),
        config={"ci": {"strict_mode": False}},
    )
    args = argparse.Namespace(
        all=False,
        trace="trace-a",
        check_consistency=False,
        format="json",
        strict=True,
        no_strict=True,
    )

    exit_code = module.cmd_verify(args, sentinel)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == module.EXIT_GENERAL_ERROR
    assert payload["exit_code"] == module.EXIT_GENERAL_ERROR
    assert payload["error"] == "conflicting strictness flags (--strict and --no-strict)"


def test_verify_stale_advisory_still_fails_on_assumption(capsys):
    module = _load_sentinel_module()
    assumptions = [
        module.AssumptionResult(id="A1", description="must hold", passed=False, message="failed")
    ]
    sentinel = _FakeSentinel(
        _make_trace_result(
            module,
            commit_status=module.Status.STALE_COMMIT,
            anchors=[],
            assumptions=assumptions,
        ),
        config={"ci": {"strict_mode": False}},
    )
    args = argparse.Namespace(
        all=False,
        trace="trace-a",
        check_consistency=False,
        format="json",
        strict=False,
        no_strict=False,
    )

    exit_code = module.cmd_verify(args, sentinel)
    payload = json.loads(capsys.readouterr().out)

    assert payload["traces"]["trace-a"]["status"] == "STALE_COMMIT"
    assert exit_code == module.EXIT_ASSUMPTION_FAILED
    assert payload["exit_code"] == module.EXIT_ASSUMPTION_FAILED


def test_route_returns_nonzero_when_auto_verify_not_grounded(capsys):
    module = _load_sentinel_module()
    sentinel = _FakeSentinel(
        _make_trace_result(module, commit_status=module.Status.STALE_COMMIT),
        config={
            "routing": [
                {
                    "symptom": "NaN in loss",
                    "trace": "trace-a",
                    "guidance": "check guards",
                }
            ]
        },
    )
    args = argparse.Namespace(symptom="NaN in loss", format="json")

    exit_code = module.cmd_route(args, sentinel)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == module.EXIT_ANCHOR_DRIFT
    assert payload["exit_code"] == module.EXIT_ANCHOR_DRIFT
    assert payload["verification"]["status"] == "STALE_COMMIT"


def test_coverage_uses_project_src_root_from_config(tmp_path, capsys):
    module = _load_sentinel_module()
    src_dir = tmp_path / "custom_src"
    src_dir.mkdir()
    (src_dir / "alpha.py").write_text("def alpha():\n    return 1\n")

    class _CoverageSentinel:
        repo_root = tmp_path
        config = {"project": {"src_root": "custom_src"}}

        @staticmethod
        def load_anchors():
            return {}

        @staticmethod
        def extract_functions(_source_path):
            return [module.FunctionInfo(name="alpha", line=1, importance="medium")]

    args = argparse.Namespace(format="json", threshold=None)
    exit_code = module.cmd_coverage(args, _CoverageSentinel())
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == module.EXIT_SUCCESS
    assert payload["summary"]["total_functions"] == 1
    assert payload["summary"]["anchored_functions"] == 0


def test_pipeline_missing_trace_is_nonzero():
    module = _load_sentinel_module()

    class _PipelineSentinel:
        repo_root = Path(".")
        config = {"ci": {"strict_mode": False}}

        @staticmethod
        def load_meta():
            return {"sentinels": {"trace-a": {}}}

        @staticmethod
        def check_consistency():
            return []

        @staticmethod
        def verify_trace(_trace_name):
            return _make_trace_result(
                module,
                commit_status=module.Status.MISSING,
                verified_commit="",
                current_commit=None,
            )

        @staticmethod
        def get_modified_files():
            return []

        @staticmethod
        def get_suggested_tests(_modified_files):
            return []

    args = argparse.Namespace(files=[], ci=False, strict=False, no_strict=False)
    exit_code = module.cmd_pipeline(args, _PipelineSentinel())
    assert exit_code == module.EXIT_GENERAL_ERROR


def test_install_adapter_unknown_adapter_returns_error(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    sentinel_dir.mkdir()
    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {"claude": {"template": "adapters/claude.skill.md", "target": "SKILL.md"}}
        },
    )
    args = argparse.Namespace(list=False, adapter="missing", force=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    assert exit_code == module.EXIT_GENERAL_ERROR
    assert "unknown adapter" in out


def test_install_adapter_missing_template_returns_error(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    sentinel_dir.mkdir()
    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {"claude": {"template": "adapters/claude.skill.md", "target": "SKILL.md"}}
        },
    )
    args = argparse.Namespace(list=False, adapter="claude", force=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    assert exit_code == module.EXIT_GENERAL_ERROR
    assert "template missing" in out


def test_install_adapter_drifted_target_requires_force(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    templates_dir = sentinel_dir / "adapters"
    templates_dir.mkdir(parents=True)
    (templates_dir / "claude.skill.md").write_text("new-template\n")
    (tmp_path / "SKILL.md").write_text("old-content\n")

    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "claude": {
                    "template": "adapters/claude.skill.md",
                    "target": "SKILL.md",
                    "mode": "copy",
                }
            }
        },
    )
    args = argparse.Namespace(list=False, adapter="claude", force=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    assert exit_code == module.EXIT_GENERAL_ERROR
    assert "Target exists and differs" in out
    assert (tmp_path / "SKILL.md").read_text() == "old-content\n"


def test_install_adapter_force_overwrites_drifted_copy_target(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    templates_dir = sentinel_dir / "adapters"
    templates_dir.mkdir(parents=True)
    (templates_dir / "claude.skill.md").write_text("new-template\n")
    (tmp_path / "SKILL.md").write_text("old-content\n")

    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "claude": {
                    "template": "adapters/claude.skill.md",
                    "target": "SKILL.md",
                    "mode": "copy",
                }
            }
        },
    )
    args = argparse.Namespace(list=False, adapter="claude", force=True)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    assert exit_code == module.EXIT_SUCCESS
    assert "Installed claude adapter" in out
    assert (tmp_path / "SKILL.md").read_text() == "new-template\n"


def test_install_adapter_list_reports_up_to_date_drifted_missing(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    templates_dir = sentinel_dir / "adapters"
    templates_dir.mkdir(parents=True)
    (templates_dir / "ok.md").write_text("same\n")
    (templates_dir / "drift.md").write_text("new\n")
    (templates_dir / "missing.md").write_text("missing\n")

    (tmp_path / "OK.md").write_text("same\n")
    (tmp_path / "DRIFT.md").write_text("old\n")
    # Missing target intentionally absent

    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "ok": {"template": "adapters/ok.md", "target": "OK.md", "mode": "copy"},
                "drift": {"template": "adapters/drift.md", "target": "DRIFT.md", "mode": "copy"},
                "missing": {
                    "template": "adapters/missing.md",
                    "target": "MISSING.md",
                    "mode": "copy",
                },
            }
        },
    )
    args = argparse.Namespace(list=True, adapter=None, force=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    assert exit_code == module.EXIT_SUCCESS
    assert "ok           [up-to-date]" in out
    assert "drift        [drifted]" in out
    assert "missing      [missing]" in out


def test_install_adapter_rejects_template_path_escape(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    sentinel_dir.mkdir()
    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "bad": {"template": "../outside.md", "target": "AGENTS.md", "mode": "copy"}
            }
        },
    )
    args = argparse.Namespace(list=False, adapter="bad", force=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    assert exit_code == module.EXIT_GENERAL_ERROR
    assert "escapes base directory" in out


def test_install_adapter_rejects_target_path_escape(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    templates_dir = sentinel_dir / "adapters"
    templates_dir.mkdir(parents=True)
    (templates_dir / "safe.md").write_text("safe\n")

    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "bad": {"template": "adapters/safe.md", "target": "../outside.md", "mode": "copy"}
            }
        },
    )
    args = argparse.Namespace(list=False, adapter="bad", force=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    assert exit_code == module.EXIT_GENERAL_ERROR
    assert "escapes base directory" in out


def test_install_adapter_managed_block_preserves_existing_target(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    templates_dir = sentinel_dir / "adapters"
    templates_dir.mkdir(parents=True)
    (templates_dir / "codex.instructions.md").write_text("sentinel-rules\n")
    (tmp_path / "AGENTS.md").write_text("global-rules\n")

    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "codex": {
                    "template": "adapters/codex.instructions.md",
                    "target": "AGENTS.md",
                    "mode": "managed_block",
                    "block_start": "<!-- sentinel:codex:start -->",
                    "block_end": "<!-- sentinel:codex:end -->",
                }
            }
        },
    )
    args = argparse.Namespace(list=False, adapter="codex", force=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    contents = (tmp_path / "AGENTS.md").read_text()

    assert exit_code == module.EXIT_SUCCESS
    assert "Installed codex adapter" in out
    assert "global-rules" in contents
    assert "<!-- sentinel:codex:start -->" in contents
    assert "sentinel-rules" in contents
    assert "<!-- sentinel:codex:end -->" in contents


def test_install_adapter_managed_block_dry_run_preserves_existing_target(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    templates_dir = sentinel_dir / "adapters"
    templates_dir.mkdir(parents=True)
    (templates_dir / "codex.instructions.md").write_text("sentinel-rules\n")
    target = tmp_path / "AGENTS.md"
    target.write_text("global-rules\n")

    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "codex": {
                    "template": "adapters/codex.instructions.md",
                    "target": "AGENTS.md",
                    "mode": "managed_block",
                    "block_start": "<!-- sentinel:codex:start -->",
                    "block_end": "<!-- sentinel:codex:end -->",
                }
            }
        },
    )
    args = argparse.Namespace(list=False, adapter="codex", force=False, dry_run=True)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out

    assert exit_code == module.EXIT_SUCCESS
    assert "Dry run: would install codex adapter" in out
    assert "appending sentinel block" in out
    assert target.read_text() == "global-rules\n"


def test_install_adapter_managed_block_create_target_no_update_note(capsys, tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = tmp_path / ".sentinel"
    templates_dir = sentinel_dir / "adapters"
    templates_dir.mkdir(parents=True)
    (templates_dir / "codex.instructions.md").write_text("sentinel-rules\n")

    sentinel = _AdapterSentinel(
        repo_root=tmp_path,
        sentinel_dir=sentinel_dir,
        config={
            "adapters": {
                "codex": {
                    "template": "adapters/codex.instructions.md",
                    "target": "AGENTS.md",
                    "mode": "managed_block",
                    "block_start": "<!-- sentinel:codex:start -->",
                    "block_end": "<!-- sentinel:codex:end -->",
                }
            }
        },
    )
    args = argparse.Namespace(list=False, adapter="codex", force=False, dry_run=False)

    exit_code = module.cmd_install_adapter(args, sentinel)
    out = capsys.readouterr().out
    contents = (tmp_path / "AGENTS.md").read_text()

    assert exit_code == module.EXIT_SUCCESS
    assert "created managed block target" in out
    assert "only sentinel-managed block content was updated" not in out
    assert "<!-- sentinel:codex:start -->" in contents


# ── v2 Phase 1 tests ──────────────────────────────────────────────────────


def test_compute_content_hash_deterministic():
    module = _load_sentinel_module()
    h1 = module._compute_content_hash("def foo():")
    h2 = module._compute_content_hash("def foo():")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_compute_content_hash_strips_whitespace():
    module = _load_sentinel_module()
    assert module._compute_content_hash("  foo  ") == module._compute_content_hash("foo")
    assert module._compute_content_hash("\tfoo\n") == module._compute_content_hash("foo")


class _FixSentinel:
    """Fake sentinel for testing cmd_fix."""

    def __init__(
        self,
        tmp_path,
        module,
        *,
        trace_status,
        anchor_impacts=None,
        source_files=None,
    ):
        self.sentinel_dir = tmp_path / ".sentinel"
        self.sentinel_dir.mkdir(exist_ok=True)
        self.repo_root = tmp_path
        self.anchors_path = self.sentinel_dir / "anchors" / "anchors.yaml"
        self.anchors_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.sentinel_dir / ".sentinel-meta.yaml"
        self.traces_dir = self.sentinel_dir / "traces"
        self.traces_dir.mkdir(exist_ok=True)
        self.config = {}
        self._module = module
        self._trace_status = trace_status
        self._anchor_impacts = anchor_impacts or []
        self._source_files = source_files or ["src/test.py"]
        self._saves = {"anchors": 0, "meta": 0}
        self._anchors_data = {}
        self._meta_data = {
            "sentinels": {
                "test-trace": {
                    "verified_commit": "aaa1111",
                    "source_files": self._source_files,
                    "status": "STALE_COMMIT",
                }
            }
        }

    def load_meta(self):
        return self._meta_data

    def load_anchors(self):
        return self._anchors_data

    def save_anchors(self, anchors):
        self._anchors_data = anchors
        self._saves["anchors"] += 1

    def save_meta(self, meta):
        self._meta_data = meta
        self._saves["meta"] += 1

    def verify_trace(self, trace_name, check_assumptions=True):  # noqa: ARG002
        return self._trace_status

    def analyze_anchor_impacts(self, trace_name):  # noqa: ARG002
        return self._anchor_impacts

    def verify_anchor(self, **kwargs):  # noqa: ARG002
        return self._module.AnchorResult(
            name="", status="VERIFIED", expected_line=1,
            actual_line=1, drift=0, message="VERIFIED"
        )

    def check_consistency(self):
        return []

    def get_current_commit(self, file_path):  # noqa: ARG002
        return "bbb2222"

    def get_modified_files(self):
        return []

    def get_suggested_tests(self, modified_files):  # noqa: ARG002
        return []


def test_fix_dry_run_no_mutations(tmp_path):
    module = _load_sentinel_module()
    trace_result = _make_trace_result(module, commit_status=module.Status.STALE_COMMIT)
    sentinel = _FixSentinel(
        tmp_path,
        module,
        trace_status=trace_result,
        anchor_impacts=[
            module.AnchorImpact(
                anchor_name="A1", status="shifted", old_line=10, new_line=15,
                suggestion="Update", content_hash_match=None,
            )
        ],
    )
    args = argparse.Namespace(trace="test-trace", all=False, dry_run=True, format="text")
    exit_code = module.cmd_fix(args, sentinel)

    assert exit_code == module.EXIT_SUCCESS
    assert sentinel._saves["anchors"] == 0
    assert sentinel._saves["meta"] == 0


def test_fix_blocks_on_deleted_anchor(tmp_path):
    module = _load_sentinel_module()
    trace_result = _make_trace_result(module, commit_status=module.Status.STALE_COMMIT)
    sentinel = _FixSentinel(
        tmp_path,
        module,
        trace_status=trace_result,
        anchor_impacts=[
            module.AnchorImpact(
                anchor_name="A1", status="deleted", old_line=10, new_line=None,
                suggestion="Pattern not found",
            )
        ],
    )
    args = argparse.Namespace(trace="test-trace", all=False, dry_run=False, format="text")
    exit_code = module.cmd_fix(args, sentinel)

    assert exit_code == module.EXIT_ANCHOR_DRIFT


def test_fix_blocks_on_content_hash_mismatch(tmp_path):
    module = _load_sentinel_module()
    trace_result = _make_trace_result(module, commit_status=module.Status.STALE_COMMIT)
    sentinel = _FixSentinel(
        tmp_path,
        module,
        trace_status=trace_result,
        anchor_impacts=[
            module.AnchorImpact(
                anchor_name="A1", status="shifted", old_line=10, new_line=15,
                suggestion="Update", content_hash_match=False,
            )
        ],
    )
    args = argparse.Namespace(trace="test-trace", all=False, dry_run=False, format="text")
    exit_code = module.cmd_fix(args, sentinel)

    assert exit_code == module.EXIT_ANCHOR_DRIFT


def test_fix_json_output(tmp_path, capsys):
    module = _load_sentinel_module()
    trace_result = _make_trace_result(module, commit_status=module.Status.VERIFIED)
    sentinel = _FixSentinel(tmp_path, module, trace_status=trace_result)
    args = argparse.Namespace(trace="test-trace", all=False, dry_run=False, format="json")
    exit_code = module.cmd_fix(args, sentinel)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == module.EXIT_SUCCESS
    assert payload["command"] == "fix"
    assert "fixes" in payload
    assert "summary" in payload


def test_gate_no_fix_skips_fix(tmp_path, capsys):
    module = _load_sentinel_module()
    trace_result = _make_trace_result(module, commit_status=module.Status.VERIFIED)
    sentinel = _FixSentinel(tmp_path, module, trace_status=trace_result)
    args = argparse.Namespace(
        no_fix=True, files=[], ci=False, strict=False, no_strict=False, format="text",
    )
    exit_code = module.cmd_gate(args, sentinel)
    out = capsys.readouterr().out

    assert exit_code == module.EXIT_SUCCESS
    assert "skipped: --no-fix" in out


def test_pipeline_emits_deprecation_warning(tmp_path, capsys):
    module = _load_sentinel_module()
    trace_result = _make_trace_result(module, commit_status=module.Status.VERIFIED)
    sentinel = _FixSentinel(tmp_path, module, trace_status=trace_result)
    args = argparse.Namespace(files=[], ci=False, strict=False, no_strict=False)
    module.cmd_pipeline(args, sentinel)
    err = capsys.readouterr().err

    assert "deprecated" in err
    assert "gate" in err


def test_verify_affected_by_filters_traces(capsys):
    module = _load_sentinel_module()
    verified_result = _make_trace_result(module, commit_status=module.Status.VERIFIED)

    class _AffectedSentinel:
        repo_root = Path("/repo")
        sentinel_dir = Path("/repo/.sentinel")
        config = {"ci": {"strict_mode": False}}
        _call_count = 0

        def load_meta(self):
            return {
                "sentinels": {
                    "trace-a": {"source_files": ["src/alpha.py"]},
                    "trace-b": {"source_files": ["src/beta.py"]},
                }
            }

        def verify_trace(self, trace_name, check_assumptions=True):  # noqa: ARG002
            self._call_count += 1
            return module.TraceVerification(
                name=trace_name,
                commit_status=module.Status.VERIFIED,
                verified_commit="abc",
                current_commit="abc",
                uncommitted_changes=False,
            )

    sentinel = _AffectedSentinel()
    args = argparse.Namespace(
        all=False, trace=None, affected_by=["src/alpha.py"],
        check_consistency=False, format="text", strict=False, no_strict=False,
    )
    exit_code = module.cmd_verify(args, sentinel)

    assert exit_code == module.EXIT_SUCCESS
    assert sentinel._call_count == 1  # Only trace-a verified


def test_verify_affected_by_no_match(capsys):
    module = _load_sentinel_module()

    class _NoMatchSentinel:
        repo_root = Path("/repo")
        sentinel_dir = Path("/repo/.sentinel")
        config = {"ci": {"strict_mode": False}}

        @staticmethod
        def load_meta():
            return {"sentinels": {"trace-a": {"source_files": ["src/alpha.py"]}}}

    args = argparse.Namespace(
        all=False, trace=None, affected_by=["src/unknown.py"],
        check_consistency=False, format="text", strict=False, no_strict=False,
    )
    exit_code = module.cmd_verify(args, _NoMatchSentinel())
    out = capsys.readouterr().out

    assert exit_code == module.EXIT_SUCCESS
    assert "No traces bound to" in out


def test_status_watch_summary_from_meta(capsys, tmp_path):
    module = _load_sentinel_module()

    class _WatchSentinel:
        sentinel_dir = tmp_path / ".sentinel"

        def __init__(self):
            self.sentinel_dir.mkdir(exist_ok=True)

        @staticmethod
        def load_meta():
            return {
                "sentinels": {
                    "trace-a": {"status": "VERIFIED"},
                    "trace-b": {"status": "STALE_COMMIT"},
                    "trace-c": {"status": "VERIFIED"},
                }
            }

    sentinel = _WatchSentinel()
    args = argparse.Namespace(watch_summary=True, format="text", verify=False)
    exit_code = module.cmd_status(args, sentinel)
    out = capsys.readouterr().out.strip()

    assert exit_code == module.EXIT_SUCCESS
    assert out == "sentinel: 2/3 verified, 1 stale"


def test_status_watch_summary_from_status_json(capsys, tmp_path):
    module = _load_sentinel_module()

    class _WatchJsonSentinel:
        sentinel_dir = tmp_path / ".sentinel"

        def __init__(self):
            self.sentinel_dir.mkdir(exist_ok=True)
            (self.sentinel_dir / "status.json").write_text(
                json.dumps({"summary": "5/5 verified"})
            )

    sentinel = _WatchJsonSentinel()
    args = argparse.Namespace(watch_summary=True, format="text", verify=False)
    exit_code = module.cmd_status(args, sentinel)
    out = capsys.readouterr().out.strip()

    assert exit_code == module.EXIT_SUCCESS
    assert out == "sentinel: 5/5 verified"


def test_status_json_written_by_fix(tmp_path):
    module = _load_sentinel_module()
    trace_result = _make_trace_result(module, commit_status=module.Status.VERIFIED)
    sentinel = _FixSentinel(tmp_path, module, trace_status=trace_result)
    args = argparse.Namespace(trace="test-trace", all=False, dry_run=False, format="text")
    module.cmd_fix(args, sentinel)

    status_path = sentinel.sentinel_dir / "status.json"
    assert status_path.exists()
    data = json.loads(status_path.read_text())
    assert "traces" in data
    assert "summary" in data
    assert "timestamp" in data
