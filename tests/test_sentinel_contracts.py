import argparse
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

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
            name="", status="VERIFIED", expected_line=1, actual_line=1, drift=0, message="VERIFIED"
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
                anchor_name="A1",
                status="shifted",
                old_line=10,
                new_line=15,
                suggestion="Update",
                content_hash_match=None,
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
                anchor_name="A1",
                status="deleted",
                old_line=10,
                new_line=None,
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
                anchor_name="A1",
                status="shifted",
                old_line=10,
                new_line=15,
                suggestion="Update",
                content_hash_match=False,
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
        no_fix=True,
        files=[],
        ci=False,
        strict=False,
        no_strict=False,
        format="text",
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
        all=False,
        trace=None,
        affected_by=["src/alpha.py"],
        check_consistency=False,
        format="text",
        strict=False,
        no_strict=False,
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
        all=False,
        trace=None,
        affected_by=["src/unknown.py"],
        check_consistency=False,
        format="text",
        strict=False,
        no_strict=False,
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
            (self.sentinel_dir / "status.json").write_text(json.dumps({"summary": "5/5 verified"}))

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


# ── v2 Phase 2a tests ──────────────────────────────────────────────────────────


# --- _FileDebouncer (4 tests) ---


def test_debouncer_fires_after_delay():
    module = _load_sentinel_module()
    debouncer = module._FileDebouncer(delay_ms=500)

    with patch.object(module.time, "monotonic") as mock_mono:
        mock_mono.return_value = 0.0
        debouncer.record("/tmp/a.py")

        mock_mono.return_value = 0.3
        assert debouncer.ready() == []

        mock_mono.return_value = 0.6
        assert debouncer.ready() == ["/tmp/a.py"]


def test_debouncer_resets_on_re_record():
    module = _load_sentinel_module()
    debouncer = module._FileDebouncer(delay_ms=500)

    with patch.object(module.time, "monotonic") as mock_mono:
        mock_mono.return_value = 0.0
        debouncer.record("/tmp/a.py")

        mock_mono.return_value = 0.4
        debouncer.record("/tmp/a.py")  # reset timer

        mock_mono.return_value = 0.6
        assert debouncer.ready() == []  # only 0.2s since re-record

        mock_mono.return_value = 1.0
        assert debouncer.ready() == ["/tmp/a.py"]


def test_debouncer_has_pending():
    module = _load_sentinel_module()
    debouncer = module._FileDebouncer(delay_ms=500)

    assert debouncer.has_pending() is False

    with patch.object(module.time, "monotonic") as mock_mono:
        mock_mono.return_value = 0.0
        debouncer.record("/tmp/a.py")
        assert debouncer.has_pending() is True

        mock_mono.return_value = 0.6
        debouncer.ready()
        assert debouncer.has_pending() is False


def test_debouncer_multiple_files_independent():
    module = _load_sentinel_module()
    debouncer = module._FileDebouncer(delay_ms=500)

    with patch.object(module.time, "monotonic") as mock_mono:
        mock_mono.return_value = 0.0
        debouncer.record("/tmp/a.py")

        mock_mono.return_value = 0.2
        debouncer.record("/tmp/b.py")

        # At t=0.6: a.py has waited 0.6s (fires), b.py only 0.4s (not yet)
        mock_mono.return_value = 0.6
        fired = debouncer.ready()
        assert fired == ["/tmp/a.py"]

        # At t=0.8: b.py has waited 0.6s (fires)
        mock_mono.return_value = 0.8
        fired = debouncer.ready()
        assert fired == ["/tmp/b.py"]


# --- build_file_trace_index (3 tests) ---


class _IndexFakeSentinel:
    """Fake sentinel for build_file_trace_index / get_monitored_dirs tests."""

    def __init__(self, repo_root, meta):
        self.repo_root = repo_root
        self._meta = meta

    def load_meta(self):
        return self._meta


def test_build_file_trace_index_maps_files_to_traces(tmp_path):
    module = _load_sentinel_module()
    # Create directory structure so Path.resolve() works
    (tmp_path / "src" / "a").mkdir(parents=True)
    (tmp_path / "src" / "b").mkdir(parents=True)
    (tmp_path / "src" / "a" / "foo.py").touch()
    (tmp_path / "src" / "b" / "bar.py").touch()

    meta = {
        "sentinels": {
            "trace-1": {"source_files": ["src/a/foo.py"]},
            "trace-2": {"source_files": ["src/a/foo.py", "src/b/bar.py"]},
        }
    }
    sentinel = _IndexFakeSentinel(tmp_path, meta)
    index = module.CodeSentinel.build_file_trace_index(sentinel)

    foo_path = str((tmp_path / "src" / "a" / "foo.py").resolve())
    bar_path = str((tmp_path / "src" / "b" / "bar.py").resolve())

    assert set(index[foo_path]) == {"trace-1", "trace-2"}
    assert index[bar_path] == ["trace-2"]


def test_get_monitored_dirs_returns_parent_dirs(tmp_path):
    module = _load_sentinel_module()
    (tmp_path / "src" / "a").mkdir(parents=True)
    (tmp_path / "src" / "b").mkdir(parents=True)
    (tmp_path / "src" / "a" / "foo.py").touch()
    (tmp_path / "src" / "b" / "bar.py").touch()

    meta = {
        "sentinels": {
            "trace-1": {"source_files": ["src/a/foo.py"]},
            "trace-2": {"source_files": ["src/b/bar.py"]},
        }
    }
    sentinel = _IndexFakeSentinel(tmp_path, meta)
    dirs = module.CodeSentinel.get_monitored_dirs(sentinel)

    expected = {
        str((tmp_path / "src" / "a").resolve()),
        str((tmp_path / "src" / "b").resolve()),
    }
    assert dirs == expected


def test_build_file_trace_index_empty_meta(tmp_path):
    module = _load_sentinel_module()
    sentinel = _IndexFakeSentinel(tmp_path, {"sentinels": {}})

    assert module.CodeSentinel.build_file_trace_index(sentinel) == {}
    assert module.CodeSentinel.get_monitored_dirs(sentinel) == set()


# --- _is_relevant_git_event (1 test) ---


def test_is_relevant_git_event_filtering(tmp_path):
    module = _load_sentinel_module()
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    def check(rel_path):
        return module._is_relevant_git_event(str((git_dir / rel_path).resolve()), tmp_path)

    # Relevant: HEAD, index, refs/heads/*
    assert check("HEAD") is True
    assert check("index") is True
    assert check("refs/heads/main") is True
    assert check("refs/heads/feature/foo") is True

    # Irrelevant: objects, logs, config, non-git paths
    assert check("objects/ab/cdef1234") is False
    assert check("logs/HEAD") is False
    assert check("config") is False
    assert module._is_relevant_git_event(str(tmp_path / "src" / "foo.py"), tmp_path) is False


# --- _handle_file_change (3 tests) ---


def _make_anchor_result(module, name, status="VERIFIED", content_hash_match=None):
    return module.AnchorResult(
        name=name,
        status=status,
        expected_line=10,
        actual_line=10,
        drift=0,
        message="ok",
        content_hash_match=content_hash_match,
    )


def _make_verification(module, name, commit_status, anchors=None):
    return module.TraceVerification(
        name=name,
        commit_status=commit_status,
        verified_commit="abc123",
        current_commit="def456",
        uncommitted_changes=False,
        anchors=anchors or [],
    )


def test_handle_file_change_verifies_only_affected_traces(tmp_path):
    module = _load_sentinel_module()
    verified = _make_verification(module, "trace-1", module.Status.VERIFIED)
    verified2 = _make_verification(module, "trace-2", module.Status.VERIFIED)

    verify_calls = []

    class _WatchSentinel:
        sentinel_dir = tmp_path / ".sentinel"

        def __init__(self):
            self.sentinel_dir.mkdir(exist_ok=True)

        def verify_trace(self, name, check_assumptions=True):  # noqa: ARG002
            verify_calls.append(name)
            return {"trace-1": verified, "trace-2": verified2}[name]

    sentinel = _WatchSentinel()
    index = {"/src/a.py": ["trace-1"], "/src/b.py": ["trace-2"]}
    cache = {"trace-1": verified, "trace-2": verified2}

    mod_name = module.__name__
    with (
        patch(f"{mod_name}._emit_watch_event"),
        patch(f"{mod_name}._write_status_json") as mock_write,
    ):
        module._handle_file_change(sentinel, ["/src/a.py"], index, False, True, cache)

    assert verify_calls == ["trace-1"]
    # _write_status_json called with full cache (both traces)
    mock_write.assert_called_once()
    written_verifications = mock_write.call_args[0][1]
    assert len(written_verifications) == 2


def test_handle_file_change_emits_fix_applied_on_auto_fix(tmp_path):
    module = _load_sentinel_module()
    stale_anchor = _make_anchor_result(module, "A1", status="DRIFT", content_hash_match=True)
    stale_verification = _make_verification(
        module, "trace-1", module.Status.STALE_COMMIT, anchors=[stale_anchor]
    )
    fixed_verification = _make_verification(module, "trace-1", module.Status.VERIFIED)

    class _WatchSentinel:
        sentinel_dir = tmp_path / ".sentinel"

        def __init__(self):
            self.sentinel_dir.mkdir(exist_ok=True)

        def verify_trace(self, name, check_assumptions=True):  # noqa: ARG002
            return stale_verification

    sentinel = _WatchSentinel()
    index = {"/src/a.py": ["trace-1"]}
    cache = {}

    fix_return = (
        0,
        [],
        [fixed_verification],
        {
            "trace-1": {
                "anchors_fixed": [{"name": "A1", "old_line": 10, "new_line": 12}],
                "new_status": "VERIFIED",
            }
        },
    )

    emitted_events = []
    mod_name = module.__name__

    def capture_event(event, json_lines):  # noqa: ARG001
        emitted_events.append(event)

    with (
        patch(f"{mod_name}._fix_traces", return_value=fix_return),
        patch(f"{mod_name}._emit_watch_event", side_effect=capture_event),
        patch(f"{mod_name}._write_status_json"),
    ):
        module._handle_file_change(sentinel, ["/src/a.py"], index, True, True, cache)

    fix_events = [e for e in emitted_events if e.event == "fix_applied"]
    assert len(fix_events) == 1
    assert fix_events[0].trace == "trace-1"
    assert fix_events[0].anchors_updated == ["A1"]
    assert fix_events[0].new_status == "VERIFIED"


def test_handle_file_change_cache_contract(tmp_path):
    module = _load_sentinel_module()
    verified_1 = _make_verification(module, "trace-1", module.Status.VERIFIED)
    stale_anchor = _make_anchor_result(module, "B1", status="DRIFT", content_hash_match=True)
    stale_2 = _make_verification(
        module, "trace-2", module.Status.STALE_COMMIT, anchors=[stale_anchor]
    )
    fixed_2 = _make_verification(module, "trace-2", module.Status.VERIFIED)

    class _WatchSentinel:
        sentinel_dir = tmp_path / ".sentinel"

        def __init__(self):
            self.sentinel_dir.mkdir(exist_ok=True)

        def verify_trace(self, name, check_assumptions=True):  # noqa: ARG002
            return stale_2

    sentinel = _WatchSentinel()
    index = {"/src/b.py": ["trace-2"]}
    cache = {"trace-1": verified_1, "trace-2": stale_2}
    original_trace1 = cache["trace-1"]

    fix_return = (
        0,
        [],
        [fixed_2],
        {
            "trace-2": {
                "anchors_fixed": [{"name": "B1", "old_line": 10, "new_line": 12}],
                "new_status": "VERIFIED",
            }
        },
    )

    mod_name = module.__name__
    with (
        patch(f"{mod_name}._fix_traces", return_value=fix_return),
        patch(f"{mod_name}._emit_watch_event"),
        patch(f"{mod_name}._write_status_json"),
    ):
        module._handle_file_change(sentinel, ["/src/b.py"], index, True, True, cache)

    # trace-1 cache entry is the SAME object (not re-verified)
    assert cache["trace-1"] is original_trace1
    # trace-2 cache entry is updated to the fixed verification
    assert cache["trace-2"].name == "trace-2"
    assert cache["trace-2"] is fixed_2


# --- Stale PID file handling (3 tests) ---


def test_stale_pidfile_cleaned_on_dead_pid(tmp_path):
    module = _load_sentinel_module()

    class _PidSentinel:
        sentinel_dir = tmp_path / ".sentinel"

    sentinel = _PidSentinel()
    sentinel.sentinel_dir.mkdir(exist_ok=True)
    pid_path = sentinel.sentinel_dir / "watch.pid"
    pid_path.write_text("99999999")

    module._check_stale_pidfile(sentinel)
    assert not pid_path.exists()


def test_stale_pidfile_blocks_on_live_pid(tmp_path):
    module = _load_sentinel_module()

    class _PidSentinel:
        sentinel_dir = tmp_path / ".sentinel"

    sentinel = _PidSentinel()
    sentinel.sentinel_dir.mkdir(exist_ok=True)
    pid_path = sentinel.sentinel_dir / "watch.pid"
    pid_path.write_text(str(os.getpid()))

    with pytest.raises(SystemExit) as exc_info:
        module._check_stale_pidfile(sentinel)
    assert exc_info.value.code == 1


def test_stale_pidfile_cleaned_on_corrupt_file(tmp_path):
    module = _load_sentinel_module()

    class _PidSentinel:
        sentinel_dir = tmp_path / ".sentinel"

    sentinel = _PidSentinel()
    sentinel.sentinel_dir.mkdir(exist_ok=True)
    pid_path = sentinel.sentinel_dir / "watch.pid"
    pid_path.write_text("not_a_pid\n")

    module._check_stale_pidfile(sentinel)
    assert not pid_path.exists()


# --- _emit_watch_event output discipline (1 test) ---


def test_emit_watch_event_json_structure(capsys):
    module = _load_sentinel_module()
    event = module.WatchEvent(
        event="fix_applied",
        timestamp="2026-03-10T00:00:00Z",
        trace="t1",
        anchors_updated=["A1"],
        new_status="VERIFIED",
    )

    # JSON Lines mode: stdout, no stderr
    module._emit_watch_event(event, json_lines=True)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert data["event"] == "fix_applied"
    assert data["trace"] == "t1"
    assert data["anchors_updated"] == ["A1"]
    assert data["new_status"] == "VERIFIED"
    # None fields must be EXCLUDED (not present as null)
    for key in ("file", "traces", "summary", "anchors_affected", "auto_fixable", "message"):
        assert key not in data

    # Human-readable mode: stderr, no stdout
    module._emit_watch_event(event, json_lines=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fixed: t1" in captured.err
