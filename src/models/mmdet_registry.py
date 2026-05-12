from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from src.models.mmdet_mask_rcnn_common import (
    load_project_experiment_config,
    resolve_project_path,
)

_MODEL_FAMILY_REGISTRY = {
    "mmdet_cascade_mask_rcnn_r50_fpn": {
        "build_module": "src.models.mmdet_cascade_mask_rcnn_r50_fpn.build",
        "mmdet_config_path": "configs/mmdetection/cascade_mask_rcnn_r50_fpn.py",
        "requires_mmpretrain": False,
    },
    "mmdet_cascade_mask_rcnn_convnext_t": {
        "build_module": "src.models.mmdet_cascade_mask_rcnn_convnext_t.build",
        "mmdet_config_path": "configs/mmdetection/cascade_mask_rcnn_convnext_t_fpn.py",
        "requires_mmpretrain": True,
    },
    "mmdet_htc_r50_fpn": {
        "build_module": "src.models.mmdet_htc_r50_fpn.build",
        "mmdet_config_path": "configs/mmdetection/htc_r50_fpn.py",
        "requires_mmpretrain": False,
    },
    "mmdet_mask_rcnn_r50_fpn": {
        "build_module": "src.models.mmdet_mask_rcnn_r50.build",
        "mmdet_config_path": "configs/mmdetection/mask_rcnn_r50_fpn.py",
        "requires_mmpretrain": False,
    },
    "mmdet_mask_rcnn_r101_fpn_coco": {
        "build_module": "src.models.mmdet_mask_rcnn_r101_coco.build",
        "mmdet_config_path": "configs/mmdetection/mask_rcnn_r101_fpn_coco.py",
        "requires_mmpretrain": False,
    },
    "mmdet_mask_rcnn_convnext_t_fpn": {
        "build_module": "src.models.mmdet_mask_rcnn_convnext_t.build",
        "mmdet_config_path": "configs/mmdetection/mask_rcnn_convnext_t_fpn.py",
        "requires_mmpretrain": True,
    },
    "mmdet_rtmdet_mask_head": {
        "build_module": "src.models.mmdet_rtmdet_mask_head.build",
        "mmdet_config_path": "configs/mmdetection/rtmdet_ins_tile_pipeline.py",
        "requires_mmpretrain": False,
    },
}


def resolve_mmdet_experiment_bundle(experiment_config: str | Path) -> dict[str, Any]:
    resolved_experiment_config = resolve_project_path(experiment_config)
    project_cfg = load_project_experiment_config(resolved_experiment_config)
    family = str(project_cfg["model"]["family"])
    family_entry = _MODEL_FAMILY_REGISTRY.get(family)
    if family_entry is None:
        supported = ", ".join(sorted(_MODEL_FAMILY_REGISTRY))
        raise ValueError(
            f"Unsupported model.family {family!r} in {resolved_experiment_config}. "
            f"Supported families: {supported}"
        )

    return {
        "experiment_config": resolved_experiment_config,
        "family": family,
        "project_cfg": project_cfg,
        "build_module": import_module(family_entry["build_module"]),
        "mmdet_config_path": resolve_project_path(family_entry["mmdet_config_path"]),
        "requires_mmpretrain": bool(family_entry["requires_mmpretrain"]),
    }


def build_mmdet_config_dict_for_experiment(experiment_config: str | Path) -> dict[str, Any]:
    experiment_bundle = resolve_mmdet_experiment_bundle(experiment_config)
    return experiment_bundle["build_module"].build_mmdet_config_dict(
        experiment_bundle["experiment_config"]
    )


def dump_resolved_config_artifacts_for_experiment(
    experiment_config: str | Path,
) -> dict[str, str]:
    experiment_bundle = resolve_mmdet_experiment_bundle(experiment_config)
    return experiment_bundle["build_module"].dump_resolved_config_artifacts(
        experiment_bundle["experiment_config"]
    )


def resolve_bridge_config_path(experiment_config: str | Path) -> Path:
    experiment_bundle = resolve_mmdet_experiment_bundle(experiment_config)
    return Path(experiment_bundle["mmdet_config_path"]).resolve()
