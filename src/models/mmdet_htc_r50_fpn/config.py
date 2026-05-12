from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.mmdet_mask_rcnn_common import (
    build_mmdet_config_dict_for_mask_rcnn,
    resolve_inference_postprocess,
)


def _build_htc_bbox_heads(num_classes: int) -> list[dict[str, Any]]:
    bbox_stds = [
        [0.1, 0.1, 0.2, 0.2],
        [0.05, 0.05, 0.1, 0.1],
        [0.033, 0.033, 0.067, 0.067],
    ]
    return [
        {
            "type": "Shared2FCBBoxHead",
            "in_channels": 256,
            "fc_out_channels": 1024,
            "roi_feat_size": 7,
            "num_classes": num_classes,
            "bbox_coder": {
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": stds,
            },
            "reg_class_agnostic": True,
            "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 1.0},
            "loss_bbox": {"type": "SmoothL1Loss", "beta": 1.0, "loss_weight": 1.0},
        }
        for stds in bbox_stds
    ]


def _build_htc_mask_heads(num_classes: int) -> list[dict[str, Any]]:
    mask_heads: list[dict[str, Any]] = []
    for stage_index in range(3):
        mask_heads.append(
            {
                "type": "HTCMaskHead",
                "num_convs": 4,
                "in_channels": 256,
                "conv_out_channels": 256,
                "num_classes": num_classes,
                "with_conv_res": stage_index > 0,
                "loss_mask": {
                    "type": "CrossEntropyLoss",
                    "use_mask": True,
                    "loss_weight": 1.0,
                },
            }
        )
    return mask_heads


def _build_htc_rcnn_train_cfg() -> dict[str, Any]:
    rcnn_train_stages = []
    for threshold in [0.5, 0.6, 0.7]:
        rcnn_train_stages.append(
            {
                "assigner": {
                    "type": "MaxIoUAssigner",
                    "pos_iou_thr": threshold,
                    "neg_iou_thr": threshold,
                    "min_pos_iou": threshold,
                    "match_low_quality": False,
                    "ignore_iof_thr": -1,
                },
                "sampler": {
                    "type": "RandomSampler",
                    "num": 512,
                    "pos_fraction": 0.25,
                    "neg_pos_ub": -1,
                    "add_gt_as_proposals": True,
                },
                "mask_size": 28,
                "pos_weight": -1,
                "debug": False,
            }
        )

    return {
        "rpn": {
            "assigner": {
                "type": "MaxIoUAssigner",
                "pos_iou_thr": 0.7,
                "neg_iou_thr": 0.3,
                "min_pos_iou": 0.3,
                "match_low_quality": True,
                "ignore_iof_thr": -1,
            },
            "sampler": {
                "type": "RandomSampler",
                "num": 256,
                "pos_fraction": 0.5,
                "neg_pos_ub": -1,
                "add_gt_as_proposals": False,
            },
            "allowed_border": 0,
            "pos_weight": -1,
            "debug": False,
        },
        "rpn_proposal": {
            "nms_pre": 2000,
            "max_per_img": 2000,
            "nms": {"type": "nms", "iou_threshold": 0.7},
            "min_bbox_size": 0,
        },
        "rcnn": rcnn_train_stages,
    }


def build_model_config(project_cfg: dict[str, Any]) -> dict[str, Any]:
    num_classes = int(project_cfg["model"]["num_classes"])
    inference_postprocess = resolve_inference_postprocess(project_cfg)
    return {
        "type": "HybridTaskCascade",
        "data_preprocessor": {
            "type": "DetDataPreprocessor",
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "bgr_to_rgb": True,
            "pad_mask": True,
            "pad_size_divisor": 32,
        },
        "backbone": {
            "type": "ResNet",
            "depth": 50,
            "num_stages": 4,
            "out_indices": (0, 1, 2, 3),
            "frozen_stages": 1,
            "norm_cfg": {"type": "BN", "requires_grad": True},
            "norm_eval": True,
            "style": "pytorch",
            "init_cfg": None,
        },
        "neck": {
            "type": "FPN",
            "in_channels": [256, 512, 1024, 2048],
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
            "type": "HybridTaskCascadeRoIHead",
            "num_stages": 3,
            "stage_loss_weights": [1.0, 0.5, 0.25],
            "interleaved": True,
            "mask_info_flow": True,
            "bbox_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {"type": "RoIAlign", "output_size": 7, "sampling_ratio": 0},
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32],
            },
            "bbox_head": _build_htc_bbox_heads(num_classes),
            "mask_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {"type": "RoIAlign", "output_size": 14, "sampling_ratio": 0},
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32],
            },
            "mask_head": _build_htc_mask_heads(num_classes),
        },
        "train_cfg": _build_htc_rcnn_train_cfg(),
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
    config = build_mmdet_config_dict_for_mask_rcnn(
        experiment_config,
        build_model_config=build_model_config,
    )
    config["load_from"] = str(config["project_cfg"]["model"]["pretrained_detector_checkpoint"])
    return config


__all__ = ["build_mmdet_config_dict", "build_model_config"]
