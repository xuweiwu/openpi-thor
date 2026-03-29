from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from pathlib import Path
import re
import tomllib
from typing import Any


SUPPORTED_COMPANION_CHECKOUT = Path("packages/openpi-thor")
HOST_LAYOUT_CHECKS = (
    Path("src/openpi"),
    Path("examples/convert_jax_model_to_pytorch.py"),
    Path("src/openpi/models_pytorch/transformers_replace"),
)
REQUIRED_OVERRIDE_DEPENDENCIES = (
    "ml-dtypes==0.5.1",
    "tensorstore==0.1.74",
)
REQUIRED_CONFLICTS = (
    (
        {"package": "openpi"},
        {"package": "openpi-thor", "group": "thor-pytorch"},
    ),
    (
        {"package": "openpi-thor"},
        {"package": "openpi-thor", "group": "thor-pytorch"},
    ),
    (
        {"package": "openpi-thor", "group": "thor-pytorch"},
        {"package": "openpi-thor", "group": "thor-lerobot"},
    ),
)
REQUIRED_LEROBOT_SOURCE = {
    "git": "https://github.com/huggingface/lerobot",
    "tag": "v0.5.0",
}


@dataclasses.dataclass
class HostPatchPlan:
    host_root: Path
    pyproject_path: Path
    changed: list[str] = dataclasses.field(default_factory=list)
    already_correct: list[str] = dataclasses.field(default_factory=list)
    could_not_patch: list[str] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    updated_text: str | None = None

    @property
    def can_write(self) -> bool:
        return not self.errors and self.updated_text is not None


def _normalize_conflict(conflict: Any) -> tuple[tuple[tuple[str, str], ...], ...]:
    if not isinstance(conflict, list | tuple):
        return ()
    normalized = []
    for item in conflict:
        if not isinstance(item, Mapping):
            return ()
        normalized.append(tuple(sorted((str(key), str(value)) for key, value in item.items())))
    return tuple(sorted(normalized))


def _format_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Mapping):
        entries = ", ".join(f"{_format_key(str(k))} = {_format_value(v)}" for k, v in value.items())
        return "{ " + entries + " }"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


def _render_tool_uv(section: Mapping[str, Any]) -> str:
    lines = ["[tool.uv]"]
    for key, value in section.items():
        if key == "conflicts":
            lines.append("conflicts = [")
            for conflict in value:
                lines.append("    [")
                for entry in conflict:
                    lines.append(f"        {_format_value(entry)},")
                lines.append("    ],")
            lines.append("]")
            continue
        lines.append(f"{key} = {_format_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def _render_tool_uv_sources(section: Mapping[str, Any]) -> str:
    lines = ["[tool.uv.sources]"]
    for key, value in section.items():
        lines.append(f"{_format_key(str(key))} = {_format_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def _find_section_span(text: str, section_name: str) -> tuple[int, int] | None:
    matches = list(re.finditer(r"(?m)^\[(?P<name>[^\]]+)\]\s*$", text))
    for index, match in enumerate(matches):
        if match.group("name") != section_name:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return match.start(), end
    return None


def _section_start(text: str, section_name: str) -> int | None:
    span = _find_section_span(text, section_name)
    return None if span is None else span[0]


def _replace_or_insert_section(
    text: str,
    section_name: str,
    rendered_section: str,
    *,
    before_sections: tuple[str, ...] = (),
    after_section: str | None = None,
) -> str:
    span = _find_section_span(text, section_name)
    block = rendered_section.strip() + "\n\n"
    if span is not None:
        start, end = span
        prefix = text[:start].rstrip()
        suffix = text[end:].lstrip("\n")
        pieces = [prefix, block.rstrip("\n"), suffix]
        return "\n\n".join(piece for piece in pieces if piece).rstrip() + "\n"

    insert_at = None
    for candidate in before_sections:
        start = _section_start(text, candidate)
        if start is not None:
            insert_at = start
            break
    if insert_at is None and after_section is not None:
        span = _find_section_span(text, after_section)
        if span is not None:
            insert_at = span[1]
    if insert_at is None:
        insert_at = len(text)

    prefix = text[:insert_at].rstrip()
    suffix = text[insert_at:].lstrip("\n")
    pieces = [prefix, block.rstrip("\n"), suffix]
    return "\n\n".join(piece for piece in pieces if piece).rstrip() + "\n"


def _merged_override_dependencies(existing: list[str], plan: HostPatchPlan) -> list[str]:
    merged = list(existing)
    missing = [item for item in REQUIRED_OVERRIDE_DEPENDENCIES if item not in merged]
    if missing:
        merged.extend(missing)
        plan.changed.append(
            "Added required tool.uv.override-dependencies entries: " + ", ".join(missing)
        )
    else:
        plan.already_correct.append("tool.uv.override-dependencies already contains the required pins")
    return merged


def _merged_conflicts(existing: list[Any], plan: HostPatchPlan) -> list[Any]:
    merged = list(existing)
    normalized_existing = {_normalize_conflict(conflict) for conflict in merged}
    added = 0
    for conflict in REQUIRED_CONFLICTS:
        normalized = _normalize_conflict([dict(item) for item in conflict])
        if normalized in normalized_existing:
            continue
        merged.append([dict(item) for item in conflict])
        normalized_existing.add(normalized)
        added += 1
    if added:
        plan.changed.append(f"Added {added} required tool.uv.conflicts entries for Thor dependency groups")
    else:
        plan.already_correct.append("tool.uv.conflicts already contains the required Thor conflict entries")
    return merged


def _merged_sources(existing: Mapping[str, Any], plan: HostPatchPlan) -> dict[str, Any]:
    merged = dict(existing)
    current = merged.get("lerobot")
    if current != REQUIRED_LEROBOT_SOURCE:
        merged["lerobot"] = dict(REQUIRED_LEROBOT_SOURCE)
        if current is None:
            plan.changed.append("Added tool.uv.sources.lerobot = official lerobot v0.5.0 tag")
        else:
            plan.changed.append("Updated tool.uv.sources.lerobot to the official lerobot v0.5.0 tag")
    else:
        plan.already_correct.append("tool.uv.sources.lerobot already points at the official lerobot v0.5.0 tag")
    return merged


def _validate_host_layout(host_root: Path, plan: HostPatchPlan) -> None:
    if not host_root.exists():
        plan.errors.append(f"Host root does not exist: {host_root}")
        return
    if not plan.pyproject_path.exists():
        plan.errors.append(f"Host root does not contain pyproject.toml: {plan.pyproject_path}")
    companion_checkout = host_root / SUPPORTED_COMPANION_CHECKOUT
    if not companion_checkout.exists():
        plan.errors.append(
            f"Expected the companion checkout at {SUPPORTED_COMPANION_CHECKOUT.as_posix()} inside the host repo."
        )
    for relative_path in HOST_LAYOUT_CHECKS:
        if not (host_root / relative_path).exists():
            plan.errors.append(
                "Host repo does not look upstream-like enough for v1 companion integration; "
                f"missing {relative_path.as_posix()}."
            )


def plan_host_pyproject_patch(host_root: str | Path) -> HostPatchPlan:
    resolved_root = Path(host_root).expanduser().resolve()
    plan = HostPatchPlan(host_root=resolved_root, pyproject_path=resolved_root / "pyproject.toml")
    _validate_host_layout(resolved_root, plan)
    if plan.errors:
        return plan

    text = plan.pyproject_path.read_text()
    try:
        pyproject = tomllib.loads(text)
    except Exception as exc:  # noqa: BLE001
        plan.errors.append(f"Failed to parse host pyproject.toml: {exc}")
        return plan

    uv = dict(pyproject.get("tool", {}).get("uv", {}))
    uv_simple = {key: value for key, value in uv.items() if key not in {"sources", "workspace"}}
    uv_simple["override-dependencies"] = _merged_override_dependencies(
        list(uv_simple.get("override-dependencies", [])),
        plan,
    )
    uv_simple["conflicts"] = _merged_conflicts(list(uv_simple.get("conflicts", [])), plan)
    sources = _merged_sources(uv.get("sources", {}), plan)

    updated_text = _replace_or_insert_section(
        text,
        "tool.uv",
        _render_tool_uv(uv_simple),
        before_sections=("tool.uv.sources", "tool.uv.workspace", "tool.ruff", "build-system"),
    )
    updated_text = _replace_or_insert_section(
        updated_text,
        "tool.uv.sources",
        _render_tool_uv_sources(sources),
        before_sections=("tool.uv.workspace", "tool.ruff", "build-system"),
        after_section="tool.uv",
    )
    plan.updated_text = updated_text
    return plan


def write_host_pyproject_patch(host_root: str | Path) -> HostPatchPlan:
    plan = plan_host_pyproject_patch(host_root)
    if not plan.can_write or plan.updated_text is None:
        return plan
    if plan.pyproject_path.read_text() != plan.updated_text:
        plan.pyproject_path.write_text(plan.updated_text)
    return plan


def companion_source_host_root(module_file: str | Path) -> Path | None:
    path = Path(module_file).resolve()
    if len(path.parents) < 3:
        return None
    if path.parents[1].name != "src":
        return None
    package_root = path.parents[2]
    if package_root.name != "openpi-thor":
        return None
    if package_root.parent.name != "packages":
        return None
    return package_root.parents[1]


def companion_source_path_warning(module_file: str | Path) -> str | None:
    path = Path(module_file).resolve()
    if len(path.parents) < 3:
        return None
    if path.parents[1].name != "src":
        return None
    package_root = path.parents[2]
    if package_root.name != "openpi-thor":
        return None
    if package_root.parent.name == "packages":
        return None
    return (
        "Source checkout detected, but v1 companion integration only supports "
        "placing openpi-thor at packages/openpi-thor inside the host openpi repo."
    )


def doctor_host_integration_warnings(module_file: str | Path) -> tuple[dict[str, Any], list[str]]:
    info: dict[str, Any] = {}
    warnings: list[str] = []

    if path_warning := companion_source_path_warning(module_file):
        warnings.append(path_warning)
        return info, warnings

    host_root = companion_source_host_root(module_file)
    if host_root is None:
        return info, warnings

    info["companion_host_root"] = str(host_root)
    info["companion_checkout"] = SUPPORTED_COMPANION_CHECKOUT.as_posix()

    plan = plan_host_pyproject_patch(host_root)
    if plan.errors:
        warnings.extend(f"Host integration check: {message}" for message in plan.errors)
        return info, warnings

    if plan.changed:
        info["companion_host_changes_needed"] = list(plan.changed)
        warnings.append(
            "Host pyproject.toml is missing some openpi-thor companion settings. "
            "Run `python packages/openpi-thor/scripts/patch_host_openpi.py --host-root . --write`."
        )
    else:
        info["companion_host_integration"] = "ok"
    return info, warnings
