from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.mmdet_mask_rcnn_common import (
    build_mask_rcnn_model_config,
    build_mmdet_config_dict_for_mask_rcnn,
    load_project_experiment_config,
)


def build_model_config(project_cfg: dict[str, Any]) -> dict[str, Any]:
    return build_mask_rcnn_model_config(
        project_cfg,
        backbone={
            "type": "ResNet",
            "depth": 50,
            "num_stages": 4,
            "out_indices": (0, 1, 2, 3),
            "frozen_stages": 1,
            "norm_cfg": {"type": "BN", "requires_grad": True},
            "norm_eval": True,
            "style": "pytorch",
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": project_cfg["model"]["pretrained"],
            },
        },
        neck={
            "type": "FPN",
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": 256,
            "num_outs": 5,
        },
    )


def build_mmdet_config_dict(experiment_config: str | Path) -> dict[str, Any]:
    return build_mmdet_config_dict_for_mask_rcnn(
        experiment_config,
        build_model_config=build_model_config,
    )


__all__ = [
    "build_mmdet_config_dict",
    "build_model_config",
    "load_project_experiment_config",
]
