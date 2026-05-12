from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.mmdet_mask_rcnn_common import (
    build_mask_rcnn_model_config,
    build_mmdet_config_dict_for_mask_rcnn,
)


def build_model_config(project_cfg: dict[str, Any]) -> dict[str, Any]:
    return build_mask_rcnn_model_config(
        project_cfg,
        backbone={
            "type": "mmpretrain.ConvNeXt",
            "arch": "tiny",
            "out_indices": [0, 1, 2, 3],
            "drop_path_rate": 0.4,
            "layer_scale_init_value": 1.0,
            "gap_before_final_norm": False,
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": project_cfg["model"]["pretrained"],
                "prefix": "backbone.",
            },
        },
        neck={
            "type": "FPN",
            "in_channels": [96, 192, 384, 768],
            "out_channels": 256,
            "num_outs": 5,
        },
    )


def build_mmdet_config_dict(experiment_config: str | Path) -> dict[str, Any]:
    return build_mmdet_config_dict_for_mask_rcnn(
        experiment_config,
        build_model_config=build_model_config,
        extra_config={
            "custom_imports": {
                "imports": ["mmpretrain.models"],
                "allow_failed_imports": False,
            }
        },
    )
