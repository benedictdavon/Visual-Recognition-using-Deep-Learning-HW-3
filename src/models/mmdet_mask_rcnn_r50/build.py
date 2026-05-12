from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config import CONFIGS_EXPERIMENTS_DIR
from src.models.mmdet_mask_rcnn_common import (
    build_mmengine_config_globals as build_mmengine_config_globals_common,
    dump_resolved_config_artifacts as dump_resolved_config_artifacts_common,
    resolve_experiment_config_path as resolve_experiment_config_path_common,
)
from src.models.mmdet_mask_rcnn_r50.config import build_mmdet_config_dict

DEFAULT_EXPERIMENT_CONFIG = CONFIGS_EXPERIMENTS_DIR / "exp001_mmdet_maskrcnn_r50.yaml"


def resolve_experiment_config_path(path: str | Path | None = None) -> Path:
    return resolve_experiment_config_path_common(
        path,
        default_experiment_config=DEFAULT_EXPERIMENT_CONFIG,
    )


def build_mmengine_config_globals(path: str | Path | None = None) -> dict[str, Any]:
    return build_mmengine_config_globals_common(
        path,
        default_experiment_config=DEFAULT_EXPERIMENT_CONFIG,
        build_mmdet_config_dict=build_mmdet_config_dict,
    )


def dump_resolved_config_artifacts(path: str | Path | None = None) -> dict[str, str]:
    return dump_resolved_config_artifacts_common(
        path,
        default_experiment_config=DEFAULT_EXPERIMENT_CONFIG,
        build_mmdet_config_dict=build_mmdet_config_dict,
    )


def main() -> int:
    artifacts = dump_resolved_config_artifacts()
    print(f"Wrote resolved config JSON to {artifacts['json']}")
    print(f"Wrote resolved config Python preview to {artifacts['python']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
