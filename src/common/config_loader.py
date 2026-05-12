from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.common.config import PROJECT_ROOT


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment specific
            raise RuntimeError(
                f"Config file {path} is not valid JSON, and PyYAML is unavailable."
            ) from error
        payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must decode to a dict.")
    return payload


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_with_bases(path: Path) -> dict[str, Any]:
    resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
    payload = _read_json_or_yaml(resolved_path)
    base_paths = payload.pop("_base_", [])
    if not isinstance(base_paths, list):
        raise ValueError(f"`_base_` in {resolved_path} must be a list.")

    merged: dict[str, Any] = {}
    for base_path in base_paths:
        base_file = (resolved_path.parent / base_path).resolve()
        merged = deep_merge_dicts(merged, load_config_with_bases(base_file))
    return deep_merge_dicts(merged, payload)
