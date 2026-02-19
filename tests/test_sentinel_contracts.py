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
    assert payload["error"] == "Specify --trace NAME or --all"


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
