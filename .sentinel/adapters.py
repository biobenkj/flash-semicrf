"""Adapter management — template installation and status checking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import CodeSentinel


def _normalize_text_for_compare(content: str) -> str:
    """Normalize text for stable comparisons across platforms."""
    return content.replace("\r\n", "\n").rstrip("\n")


def _resolve_path_within(
    base_dir: Path, configured_path: str, field_name: str
) -> tuple[Path | None, str | None]:
    """Resolve a configured path and ensure it stays under base_dir."""
    if not configured_path:
        return None, f"{field_name} is empty"

    raw = Path(configured_path)
    if raw.is_absolute():
        return None, f"{field_name} must be relative, got absolute path: {configured_path}"

    resolved = (base_dir / raw).resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        return None, f"{field_name} escapes base directory: {configured_path}"

    return resolved, None


def _managed_block_markers(spec: dict, adapter_name: str) -> tuple[str, str]:
    """Get managed block markers for an adapter."""
    start = spec.get("block_start", f"<!-- sentinel:{adapter_name}:start -->")
    end = spec.get("block_end", f"<!-- sentinel:{adapter_name}:end -->")
    return start, end


def _extract_managed_block_content(content: str, block_start: str, block_end: str) -> str | None:
    """Extract managed block body, or None if markers are missing."""
    start_idx = content.find(block_start)
    if start_idx < 0:
        return None
    end_idx = content.find(block_end, start_idx + len(block_start))
    if end_idx < 0:
        return None
    return content[start_idx + len(block_start) : end_idx].strip("\r\n")


def _render_managed_block(template_content: str, block_start: str, block_end: str) -> str:
    """Render a managed block wrapper around template content."""
    return f"{block_start}\n{template_content.rstrip()}\n{block_end}\n"


def _prepare_adapter_install(
    sentinel: CodeSentinel, adapter_name: str, spec: dict
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate adapter config and resolve safe paths."""
    template_rel = spec.get("template", "")
    target_rel = spec.get("target", "")
    mode = spec.get("mode", "copy")

    if mode not in {"copy", "managed_block"}:
        return None, f"adapter '{adapter_name}' has unsupported mode: {mode}"

    template_path, template_err = _resolve_path_within(
        sentinel.sentinel_dir, template_rel, f"adapter '{adapter_name}' template"
    )
    if template_err:
        return None, template_err

    target_path, target_err = _resolve_path_within(
        sentinel.repo_root, target_rel, f"adapter '{adapter_name}' target"
    )
    if target_err:
        return None, target_err

    block_start, block_end = _managed_block_markers(spec, adapter_name)
    return {
        "name": adapter_name,
        "mode": mode,
        "template_rel": template_rel,
        "target_rel": target_rel,
        "template_path": template_path,
        "target_path": target_path,
        "block_start": block_start,
        "block_end": block_end,
    }, None


def _adapter_install_status(adapter_cfg: dict[str, Any]) -> tuple[str, str]:
    """Return adapter status: up_to_date, drifted, missing, invalid."""
    template_path: Path = adapter_cfg["template_path"]
    target_path: Path = adapter_cfg["target_path"]

    if not template_path.exists():
        return "invalid", f"template missing: {adapter_cfg['template_rel']}"
    if not template_path.is_file():
        return "invalid", f"template is not a file: {adapter_cfg['template_rel']}"

    template_content = template_path.read_text()
    if not target_path.exists():
        return "missing", "target file does not exist"

    target_content = target_path.read_text()
    mode = adapter_cfg["mode"]
    if mode == "copy":
        if _normalize_text_for_compare(target_content) == _normalize_text_for_compare(
            template_content
        ):
            return "up_to_date", "target matches template"
        return "drifted", "target differs from template"

    managed_content = _extract_managed_block_content(
        target_content, adapter_cfg["block_start"], adapter_cfg["block_end"]
    )
    if managed_content is None:
        return "drifted", "managed block markers not found in target"
    if _normalize_text_for_compare(managed_content) == _normalize_text_for_compare(
        template_content
    ):
        return "up_to_date", "managed block matches template"
    return "drifted", "managed block differs from template"
