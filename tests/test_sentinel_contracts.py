import argparse
import importlib.util
import json
import os
import subprocess
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


# ══════════════════════════════════════════════════════════════════════
# Phase 2b: init --scan tests
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo with Python source files for scan testing.

    Files:
      src/pkg/core.py     - 2 critical functions (forward + backward) -> above threshold
      src/pkg/multi.py    - 2 classes each with forward() -> tests disambiguation
      src/pkg/utils.py    - 1 function (helper) matching no critical/high patterns
                            (intentionally below min_functions=2 threshold)
      src/pkg/__init__.py - excluded by preset exclude_globs
    """
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "core.py").write_text(
        "def forward(x):\n    return x\n\ndef backward(x):\n    return x\n"
    )
    (src / "multi.py").write_text(
        "class A:\n    def forward(self, x):\n        return x\n\n"
        "class B:\n    def forward(self, x):\n        return x\n"
    )
    # utils.py has only helper() — matches no critical/high patterns.
    # Intentionally below min_functions=2 threshold.
    (src / "utils.py").write_text("def helper():\n    pass\n")
    (src / "__init__.py").write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    return tmp_path


def _make_real_sentinel(module, repo_root, src_root="src/pkg"):
    """Bootstrap .sentinel/ and return a real CodeSentinel instance."""
    sentinel_dir = module.bootstrap_sentinel_dir(repo_root)
    (sentinel_dir / "sentinel.yaml").write_text(
        f'version: "1.0"\nproject:\n  name: test\n  src_root: {src_root}\n'
    )
    return module.CodeSentinel(sentinel_dir)


# ── 1. ScanPreset + built-in presets ─────────────────────────────────


def test_builtin_presets_are_well_formed():
    module = _load_sentinel_module()
    assert set(module.BUILTIN_PRESETS.keys()) == {"default", "pytorch", "typescript"}
    for name, preset in module.BUILTIN_PRESETS.items():
        assert len(preset.file_globs) > 0, f"{name} has no file_globs"
        assert preset.language, f"{name} has empty language"
        assert preset.min_functions >= 1, f"{name} has min_functions < 1"
    assert module.BUILTIN_PRESETS["default"].language == "python"
    assert module.BUILTIN_PRESETS["pytorch"].language == "python"
    assert module.BUILTIN_PRESETS["typescript"].language == "typescript"


def test_default_preset_min_functions_is_two():
    module = _load_sentinel_module()
    assert module.BUILTIN_PRESETS["default"].min_functions == 2


# ── 2. derive_trace_name ─────────────────────────────────────────────


def test_derive_trace_name_basic_stem(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    p1 = Path(tmp_path / "src" / "genmul.py")
    p1.parent.mkdir(parents=True, exist_ok=True)
    assert sentinel.derive_trace_name(p1, set()) == "genmul"
    p2 = Path(tmp_path / "src" / "triton_forward.py")
    assert sentinel.derive_trace_name(p2, set()) == "triton-forward"


def test_derive_trace_name_strips_leading_hyphens(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    # _genbmm dir -> parent becomes "genbmm" not "-genbmm"
    p = Path(tmp_path / "src" / "_genbmm" / "genmul.py")
    p.parent.mkdir(parents=True, exist_ok=True)
    # No collision: stem "genmul" used directly
    assert sentinel.derive_trace_name(p, set()) == "genmul"
    # Force collision on stem -> parent prefix should not have leading hyphen
    assert sentinel.derive_trace_name(p, {"genmul"}) == "genbmm-genmul"


def test_derive_trace_name_collision_uses_parent(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    p1 = Path(tmp_path / "streaming" / "forward.py")
    p2 = Path(tmp_path / "batch" / "forward.py")
    existing = set()
    name1 = sentinel.derive_trace_name(p1, existing)
    assert name1 == "forward"
    existing.add(name1)
    name2 = sentinel.derive_trace_name(p2, existing)
    assert name2 == "batch-forward"


def test_derive_trace_name_collision_with_existing_trace(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    p = Path(tmp_path / "src" / "triton_forward.py")
    # "triton-forward" already taken by a manually-created trace
    name = sentinel.derive_trace_name(p, {"triton-forward"})
    assert name != "triton-forward"
    assert len(name) > 0


def test_derive_trace_name_numeric_suffix_last_resort(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    p = Path(tmp_path / "streaming" / "forward.py")
    # Both stem and parent-qualified are taken
    name = sentinel.derive_trace_name(p, {"forward", "streaming-forward"})
    assert name == "forward-2"


def test_derive_trace_name_unicode_does_not_crash(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    p = Path(tmp_path / "src" / "señal_procesador.py")
    name = sentinel.derive_trace_name(p, set())
    assert isinstance(name, str)
    assert len(name) > 0


# ── 3. scaffold_trace ────────────────────────────────────────────────


def test_scaffold_trace_returns_verified_status(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "mod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def forward(x):\n    return x\n\ndef backward(x):\n    return x\n")

    trace_md, anchor_dict, meta_entry = sentinel.scaffold_trace(
        src, "mod", "abc1234", status="VERIFIED"
    )
    assert "**Status:** VERIFIED" in trace_md
    assert "abc1234" in trace_md
    assert meta_entry["status"] == "VERIFIED"
    assert meta_entry["verified_commit"] == "abc1234"


def test_scaffold_trace_draft_mode_backward_compat(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "mod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def forward(x):\n    return x\n")

    trace_md, _, meta_entry = sentinel.scaffold_trace(src, "mod", "TODO", status="DRAFT")
    assert "**Status:** DRAFT" in trace_md
    assert "TODO" in trace_md
    assert meta_entry["status"] == "DRAFT"


def test_scaffold_trace_computes_content_hashes(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "mod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    content = "def forward(x):\n    return x\n\ndef backward(x):\n    return x\n"
    src.write_text(content)
    lines = content.splitlines()

    _, anchor_dict, _ = sentinel.scaffold_trace(src, "mod", "abc", status="VERIFIED")
    assert len(anchor_dict) > 0
    for aname, adef in anchor_dict.items():
        assert adef["content_hash"], f"anchor {aname} has empty content_hash"
        expected_hash = module._compute_content_hash(lines[adef["expected_line"] - 1])
        assert adef["content_hash"] == expected_hash, f"hash mismatch for {aname}"


def test_scaffold_trace_disambiguates_duplicate_function_names(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "multi.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "class A:\n    def forward(self, x):\n        return x\n\n"
        "class B:\n    def forward(self, x):\n        return x\n"
    )

    _, anchor_dict, _ = sentinel.scaffold_trace(src, "multi", "abc", status="VERIFIED")
    keys = list(anchor_dict.keys())
    # First occurrence keeps unqualified name, second gets class-prefixed (first-wins)
    assert "FORWARD" in keys, f"expected FORWARD in {keys}"
    assert any(
        k != "FORWARD" and "FORWARD" in k for k in keys
    ), f"expected a disambiguated FORWARD variant in {keys}"
    # Both should have 'after' field (both are ambiguous)
    for aname, adef in anchor_dict.items():
        if "FORWARD" in aname:
            assert "after" in adef, f"anchor {aname} missing 'after' for disambiguation"


def test_scaffold_trace_no_anchors_for_medium_only_functions(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "boring.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def helper():\n    pass\n\ndef another():\n    pass\n")

    _, anchor_dict, meta_entry = sentinel.scaffold_trace(src, "boring", "abc", status="VERIFIED")
    assert len(anchor_dict) == 0
    assert meta_entry["anchors"] == []


def test_scaffold_trace_limits_to_ten_anchors(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "many.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    # 15 functions matching the critical pattern "def forward"
    lines = []
    for i in range(15):
        lines.append(f"def forward_{i}(x):\n    return x\n")
    src.write_text("\n".join(lines))
    # Override critical patterns to match all forward_N
    _, anchor_dict, _ = sentinel.scaffold_trace(
        src,
        "many",
        "abc",
        status="VERIFIED",
        critical_patterns=[r"def forward_\d+\("],
    )
    assert len(anchor_dict) <= 10


def test_scaffold_trace_empty_file(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "empty.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("")

    trace_md, anchor_dict, meta_entry = sentinel.scaffold_trace(
        src, "empty", "abc", status="VERIFIED"
    )
    assert len(anchor_dict) == 0
    assert "# Sentinel: empty" in trace_md
    assert meta_entry["anchors"] == []


def test_scaffold_trace_binary_content(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    src = tmp_path / "src" / "pkg" / "binary.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"\x00\x01\x02\xff\xfe")

    # Should not crash — extract_functions handles non-text gracefully
    try:
        _, anchor_dict, _ = sentinel.scaffold_trace(src, "binary", "abc", status="VERIFIED")
        assert len(anchor_dict) == 0
    except UnicodeDecodeError:
        # Acceptable — binary files may raise on read_text()
        pass


# ── 4. load_scan_preset ──────────────────────────────────────────────


def test_load_scan_preset_default_uses_config_src_root(tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = module.bootstrap_sentinel_dir(tmp_path)
    (sentinel_dir / "sentinel.yaml").write_text(
        'version: "1.0"\nproject:\n  name: test\n  src_root: lib/mypackage\n'
    )
    sentinel = module.CodeSentinel(sentinel_dir)
    preset = sentinel.load_scan_preset("default")
    assert preset.file_globs == ["lib/mypackage/**/*.py"]


def test_load_scan_preset_default_uses_config_init_patterns(tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = module.bootstrap_sentinel_dir(tmp_path)
    (sentinel_dir / "sentinel.yaml").write_text(
        'version: "1.0"\nproject:\n  name: test\n  src_root: src\n'
        "init_patterns:\n  critical:\n    - 'def custom_critical\\('\n"
    )
    sentinel = module.CodeSentinel(sentinel_dir)
    preset = sentinel.load_scan_preset("default")
    assert "def custom_critical\\(" in preset.critical_patterns


def test_load_scan_preset_pytorch_merges_extra_patterns(tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = module.bootstrap_sentinel_dir(tmp_path)
    (sentinel_dir / "sentinel.yaml").write_text(
        'version: "1.0"\nproject:\n  name: test\n  src_root: src\n'
    )
    sentinel = module.CodeSentinel(sentinel_dir)
    preset = sentinel.load_scan_preset("pytorch")
    # Should have pytorch-specific extras
    assert any("torch" in p for p in preset.critical_patterns)
    assert any("training_step" in p for p in preset.high_patterns)


def test_load_scan_preset_unknown_raises_valueerror(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    with pytest.raises(ValueError, match="Unknown preset"):
        sentinel.load_scan_preset("nonexistent")


def test_load_scan_preset_custom_from_presets_dir(tmp_path):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    presets_dir = sentinel.sentinel_dir / "presets"
    presets_dir.mkdir(exist_ok=True)
    (presets_dir / "custom.yaml").write_text(
        "name: custom\n"
        "description: test preset\n"
        "language: python\n"
        "file_globs: ['src/**/*.py']\n"
        "exclude_globs: []\n"
        "critical_patterns: ['def main\\(']\n"
        "high_patterns: []\n"
        "min_functions: 1\n"
    )
    preset = sentinel.load_scan_preset("custom")
    assert preset.name == "custom"
    assert preset.language == "python"
    assert preset.min_functions == 1


# ── 5. discover_source_files ─────────────────────────────────────────


def test_discover_source_files_skips_unsupported_language(tmp_path, capsys):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, tmp_path)
    ts_preset = module.BUILTIN_PRESETS["typescript"]
    result = sentinel.discover_source_files(ts_preset)
    assert result == []
    assert "no function extractor" in capsys.readouterr().err


def test_discover_source_files_fails_gracefully_without_git(tmp_path):
    module = _load_sentinel_module()
    # tmp_path is NOT a git repo — no git init
    sentinel = _make_real_sentinel(module, tmp_path)
    preset = module.BUILTIN_PRESETS["default"]
    # Override to use "python" language but run in non-git dir
    result = sentinel.discover_source_files(preset)
    assert result == []  # Should return empty, not crash


# ── 6. bootstrap_sentinel_dir ────────────────────────────────────────


def test_bootstrap_creates_directory_structure(tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = module.bootstrap_sentinel_dir(tmp_path)
    for subdir in ["traces", "anchors", "adapters", "presets", "diffs", "hooks"]:
        assert (sentinel_dir / subdir).is_dir(), f"missing {subdir}/"


def test_bootstrap_creates_config_files(tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = module.bootstrap_sentinel_dir(tmp_path)
    import yaml

    meta = yaml.safe_load((sentinel_dir / ".sentinel-meta.yaml").read_text())
    assert "sentinels" in meta
    anchors = yaml.safe_load((sentinel_dir / "anchors" / "anchors.yaml").read_text())
    assert anchors is not None or anchors == {}
    config = yaml.safe_load((sentinel_dir / "sentinel.yaml").read_text())
    assert "project" in config


def test_bootstrap_writes_verify_anchor_sh(tmp_path):
    module = _load_sentinel_module()
    sentinel_dir = module.bootstrap_sentinel_dir(tmp_path)
    script = sentinel_dir / "anchors" / "verify-anchor.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111, "verify-anchor.sh not executable"
    assert script.read_text().startswith("#!/bin/bash")


def test_bootstrap_idempotent_preserves_existing_meta(tmp_path):
    module = _load_sentinel_module()
    # First bootstrap
    sentinel_dir = module.bootstrap_sentinel_dir(tmp_path)
    # Write custom meta
    meta_path = sentinel_dir / ".sentinel-meta.yaml"
    meta_path.write_text("version: '2.0'\nsentinels:\n  custom-trace:\n    status: VERIFIED\n")
    original_meta = meta_path.read_text()
    # Second bootstrap
    module.bootstrap_sentinel_dir(tmp_path)
    # Meta should NOT be overwritten
    assert meta_path.read_text() == original_meta
    # But verify-anchor.sh IS always overwritten (stays current)
    assert (sentinel_dir / "anchors" / "verify-anchor.sh").exists()


# ── 7. cmd_init_scan (integration with git_repo + real CodeSentinel) ─


def test_init_scan_dry_run_writes_nothing(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=True, force=False)
    exit_code = module.cmd_init_scan(args, sentinel)
    assert exit_code == module.EXIT_SUCCESS
    # No trace files should be written
    traces = list(sentinel.traces_dir.glob("*.md"))
    assert len(traces) == 0
    # Meta should still be empty (only bootstrapped defaults)
    meta = sentinel.load_meta()
    assert len(meta.get("sentinels", {})) == 0


def test_init_scan_creates_traces_and_verifies(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    exit_code = module.cmd_init_scan(args, sentinel)
    assert exit_code == module.EXIT_SUCCESS

    # Should have created traces
    meta = sentinel.load_meta()
    sentinels = meta.get("sentinels", {})
    assert len(sentinels) > 0

    # Traces should exist as files
    for name in sentinels:
        trace_path = sentinel.traces_dir / f"{name}.md"
        assert trace_path.exists(), f"trace file missing: {trace_path}"

    # status.json should exist and reflect reality
    import json

    status_path = sentinel.sentinel_dir / "status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text())
    assert "traces" in status
    assert "summary" in status


def test_init_scan_skips_already_covered_files(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    # Pre-populate meta with core.py covered
    meta = sentinel.load_meta()
    meta.setdefault("sentinels", {})["existing-core"] = {
        "verified_commit": "aaa1111",
        "source_files": ["src/pkg/core.py"],
        "anchors": [],
        "status": "VERIFIED",
    }
    sentinel.save_meta(meta)

    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    module.cmd_init_scan(args, sentinel)

    meta = sentinel.load_meta()
    # core.py should NOT have a second sentinel entry
    core_sentinels = [
        name
        for name, s in meta["sentinels"].items()
        if "src/pkg/core.py" in s.get("source_files", [])
    ]
    assert len(core_sentinels) == 1
    assert core_sentinels[0] == "existing-core"


def test_init_scan_idempotent_second_run(git_repo, capsys):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    module.cmd_init_scan(args, sentinel)
    first_meta = sentinel.load_meta()
    first_count = len(first_meta.get("sentinels", {}))

    # Second run
    capsys.readouterr()  # clear
    module.cmd_init_scan(args, sentinel)
    captured = capsys.readouterr()

    second_meta = sentinel.load_meta()
    assert len(second_meta.get("sentinels", {})) == first_count
    assert "No new traces created" in captured.out


def test_init_scan_merges_with_existing_anchors_and_meta(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)

    # Pre-populate with an existing trace
    anchors = {
        "existing-trace": {
            "MY_ANCHOR": {
                "file": "other.py",
                "pattern": "x",
                "expected_line": 1,
                "drift_tolerance": 20,
            }
        }
    }
    sentinel.save_anchors(anchors)
    meta = sentinel.load_meta()
    meta.setdefault("sentinels", {})["existing-trace"] = {
        "verified_commit": "old123",
        "source_files": ["other.py"],
        "anchors": ["MY_ANCHOR"],
        "status": "VERIFIED",
    }
    sentinel.save_meta(meta)

    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    module.cmd_init_scan(args, sentinel)

    # Existing entry must be preserved
    final_meta = sentinel.load_meta()
    assert "existing-trace" in final_meta["sentinels"]
    assert final_meta["sentinels"]["existing-trace"]["verified_commit"] == "old123"

    final_anchors = sentinel.load_anchors()
    assert "existing-trace" in final_anchors
    assert "MY_ANCHOR" in final_anchors["existing-trace"]


def test_init_scan_force_regenerates_covered_files(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    # First run
    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    module.cmd_init_scan(args, sentinel)
    first_meta = sentinel.load_meta()
    first_count = len(first_meta.get("sentinels", {}))

    # Force re-run
    args_force = argparse.Namespace(
        scan=True, preset="default", adapter=None, dry_run=False, force=True
    )
    module.cmd_init_scan(args_force, sentinel)
    force_meta = sentinel.load_meta()
    # Should have same or more entries (force regenerates, doesn't remove)
    assert len(force_meta.get("sentinels", {})) >= first_count


def test_init_scan_respects_min_functions_threshold(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    module.cmd_init_scan(args, sentinel)

    meta = sentinel.load_meta()
    # utils.py has only helper() (no critical/high matches) -> should NOT have a trace
    for name, entry in meta.get("sentinels", {}).items():
        source_files = entry.get("source_files", [])
        assert not any(
            "utils.py" in sf for sf in source_files
        ), f"utils.py should have been skipped (below threshold) but found in {name}"


def test_init_scan_status_json_reflects_reality(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    module.cmd_init_scan(args, sentinel)

    import json

    status = json.loads((sentinel.sentinel_dir / "status.json").read_text())
    meta = sentinel.load_meta()
    # status.json should have an entry for every sentinel in meta
    for name in meta.get("sentinels", {}):
        assert name in status["traces"], f"trace {name} missing from status.json"


def test_init_scan_no_discoverable_files(git_repo, capsys):
    module = _load_sentinel_module()
    # Point src_root to an empty directory
    empty_src = git_repo / "empty_src"
    empty_src.mkdir()
    sentinel = _make_real_sentinel(module, git_repo, src_root="empty_src")
    args = argparse.Namespace(scan=True, preset="default", adapter=None, dry_run=False, force=False)
    exit_code = module.cmd_init_scan(args, sentinel)
    assert exit_code == module.EXIT_SUCCESS
    assert "No source files found" in capsys.readouterr().out


def test_init_scan_with_adapter_installs_skill(git_repo):
    module = _load_sentinel_module()
    sentinel = _make_real_sentinel(module, git_repo)
    # Set up adapter config
    import yaml

    config = yaml.safe_load((sentinel.sentinel_dir / "sentinel.yaml").read_text())
    config["adapters"] = {
        "claude": {
            "template": "adapters/claude.skill.md",
            "target": ".claude/skills/code-sentinel/SKILL.md",
            "description": "Claude adapter",
        }
    }
    (sentinel.sentinel_dir / "sentinel.yaml").write_text(
        yaml.dump(config, default_flow_style=False)
    )
    sentinel.config = sentinel.load_config()
    # Create template
    adapters_dir = sentinel.sentinel_dir / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    (adapters_dir / "claude.skill.md").write_text("# Sentinel Skill\n")

    args = argparse.Namespace(
        scan=True, preset="default", adapter="claude", dry_run=False, force=False
    )
    module.cmd_init_scan(args, sentinel)

    # Adapter target should exist
    target = git_repo / ".claude" / "skills" / "code-sentinel" / "SKILL.md"
    assert target.exists(), "adapter skill file was not installed"


# ── 8. cmd_init dispatch ─────────────────────────────────────────────


def test_init_no_source_no_scan_prints_error(capsys):
    module = _load_sentinel_module()
    sentinel = _FakeSentinel(
        _make_trace_result(module, commit_status=module.Status.VERIFIED),
    )
    args = argparse.Namespace(
        source=None,
        scan=False,
        name=None,
        force=False,
        preset="default",
        adapter=None,
        dry_run=False,
    )
    exit_code = module.cmd_init(args, sentinel)
    assert exit_code == module.EXIT_GENERAL_ERROR
    captured = capsys.readouterr()
    assert "source file required" in captured.out
