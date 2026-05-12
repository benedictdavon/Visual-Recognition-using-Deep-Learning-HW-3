from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.mmdet_cascade_mask_rcnn_r50_fpn.config import (
    _build_cascade_bbox_heads,
    _build_cascade_rcnn_train_cfg,
)
from src.models.mmdet_mask_rcnn_common import (
    build_mmdet_config_dict_for_mask_rcnn,
    resolve_inference_postprocess,
)


def build_model_config(project_cfg: dict[str, Any]) -> dict[str, Any]:
    num_classes = int(project_cfg["model"]["num_classes"])
    inference_postprocess = resolve_inference_postprocess(project_cfg)
    return {
        "type": "CascadeRCNN",
        "data_preprocessor": {
            "type": "DetDataPreprocessor",
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "bgr_to_rgb": True,
            "pad_mask": True,
            "pad_size_divisor": 32,
        },
        "backbone": {
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
        "neck": {
            "type": "FPN",
            "in_channels": [96, 192, 384, 768],
            "out_channels": 256,
            "num_outs": 5,
        },
        "rpn_head": {
            "type": "RPNHead",
            "in_channels": 256,
            "feat_channels": 256,
            "anchor_generator": {
                "type": "AnchorGenerator",
                "scales": [8],
                "ratios": [0.5, 1.0, 2.0],
                "strides": [4, 8, 16, 32, 64],
            },
            "bbox_coder": {
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [1.0, 1.0, 1.0, 1.0],
            },
            "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": True, "loss_weight": 1.0},
            "loss_bbox": {"type": "SmoothL1Loss", "beta": 1.0 / 9.0, "loss_weight": 1.0},
        },
        "roi_head": {
            "type": "CascadeRoIHead",
            "num_stages": 3,
            "stage_loss_weights": [1.0, 0.5, 0.25],
            "bbox_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {"type": "RoIAlign", "output_size": 7, "sampling_ratio": 0},
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32],
            },
            "bbox_head": _build_cascade_bbox_heads(num_classes),
            "mask_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {"type": "RoIAlign", "output_size": 14, "sampling_ratio": 0},
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32],
            },
            "mask_head": {
                "type": "FCNMaskHead",
                "num_convs": 4,
                "in_channels": 256,
                "conv_out_channels": 256,
                "num_classes": num_classes,
                "loss_mask": {"type": "CrossEntropyLoss", "use_mask": True, "loss_weight": 1.0},
            },
        },
        "train_cfg": _build_cascade_rcnn_train_cfg(),
        "test_cfg": {
            "rpn": {
                "nms_pre": 1000,
                "max_per_img": 1000,
                "nms": {"type": "nms", "iou_threshold": 0.7},
                "min_bbox_size": 0,
            },
            "rcnn": {
                "score_thr": inference_postprocess["score_thr"],
                "nms": inference_postprocess["nms"],
                "max_per_img": inference_postprocess["max_per_img"],
                "mask_thr_binary": 0.5,
            },
        },
    }


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


__all__ = ["build_mmdet_config_dict", "build_model_config"]
