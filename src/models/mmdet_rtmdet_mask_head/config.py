from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.mmdet_mask_rcnn_common import (
    build_mmdet_config_dict_for_mask_rcnn,
    resolve_inference_postprocess,
)


_RTMDET_INS_SIZE_SPECS: dict[str, dict[str, Any]] = {
    "tiny": {
        "deepen_factor": 0.167,
        "widen_factor": 0.375,
        "neck_in_channels": [96, 192, 384],
        "neck_out_channels": 96,
        "num_csp_blocks": 1,
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth",
    },
    "s": {
        "deepen_factor": 0.33,
        "widen_factor": 0.5,
        "neck_in_channels": [128, 256, 512],
        "neck_out_channels": 128,
        "num_csp_blocks": 1,
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
    },
    "m": {
        "deepen_factor": 0.67,
        "widen_factor": 0.75,
        "neck_in_channels": [192, 384, 768],
        "neck_out_channels": 192,
        "num_csp_blocks": 2,
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth",
    },
    "l": {
        "deepen_factor": 1.0,
        "widen_factor": 1.0,
        "neck_in_channels": [256, 512, 1024],
        "neck_out_channels": 256,
        "num_csp_blocks": 3,
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_l_8xb32-300e_coco/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth",
    },
    "x": {
        "deepen_factor": 1.33,
        "widen_factor": 1.25,
        "neck_in_channels": [320, 640, 1280],
        "neck_out_channels": 320,
        "num_csp_blocks": 4,
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_x_8xb16-300e_coco/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth",
    },
}


def _resolve_model_size(project_cfg: dict[str, Any]) -> str:
    model_size = str(project_cfg["model"].get("model_size", "s")).lower()
    if model_size not in _RTMDET_INS_SIZE_SPECS:
        supported = ", ".join(sorted(_RTMDET_INS_SIZE_SPECS))
        raise ValueError(
            f"Unsupported RTMDet-Ins model_size {model_size!r}. Supported sizes: {supported}."
        )
    return model_size


def _resolve_pretrained_checkpoint(project_cfg: dict[str, Any], model_size: str) -> str:
    configured = project_cfg["model"].get("pretrained_detector_checkpoint")
    if configured:
        return str(configured)
    return str(_RTMDET_INS_SIZE_SPECS[model_size]["checkpoint"])


def build_model_config(project_cfg: dict[str, Any]) -> dict[str, Any]:
    num_classes = int(project_cfg["model"]["num_classes"])
    model_size = _resolve_model_size(project_cfg)
    size_spec = _RTMDET_INS_SIZE_SPECS[model_size]
    head_channels = int(size_spec["neck_out_channels"])
    assigner_topk = int(project_cfg["model"].get("assigner_topk", 13))
    if assigner_topk <= 0:
        raise ValueError(f"model.assigner_topk must be positive, got {assigner_topk!r}.")
    inference_postprocess = resolve_inference_postprocess(project_cfg)
    norm_cfg = {"type": "BN", "requires_grad": True}
    act_cfg = {"type": "SiLU", "inplace": True}

    return {
        "type": "RTMDet",
        "data_preprocessor": {
            "type": "DetDataPreprocessor",
            "mean": [103.53, 116.28, 123.675],
            "std": [57.375, 57.12, 58.395],
            "bgr_to_rgb": False,
            "pad_mask": True,
            "pad_size_divisor": 32,
            "batch_augments": None,
        },
        "backbone": {
            "type": "CSPNeXt",
            "arch": "P5",
            "expand_ratio": 0.5,
            "deepen_factor": float(size_spec["deepen_factor"]),
            "widen_factor": float(size_spec["widen_factor"]),
            "out_indices": (2, 3, 4),
            "frozen_stages": -1,
            "use_depthwise": False,
            "channel_attention": True,
            "norm_cfg": norm_cfg,
            "act_cfg": act_cfg,
            "init_cfg": None,
        },
        "neck": {
            "type": "CSPNeXtPAFPN",
            "in_channels": [int(channel) for channel in size_spec["neck_in_channels"]],
            "out_channels": head_channels,
            "num_csp_blocks": int(size_spec["num_csp_blocks"]),
            "expand_ratio": 0.5,
            "norm_cfg": norm_cfg,
            "act_cfg": act_cfg,
        },
        "bbox_head": {
            "type": "RTMDetInsSepBNHead",
            "num_classes": num_classes,
            "in_channels": head_channels,
            "stacked_convs": 2,
            "share_conv": True,
            "pred_kernel_size": 1,
            "feat_channels": head_channels,
            "act_cfg": act_cfg,
            "norm_cfg": norm_cfg,
            "anchor_generator": {
                "type": "MlvlPointGenerator",
                "offset": 0,
                "strides": [8, 16, 32],
            },
            "bbox_coder": {"type": "DistancePointBBoxCoder"},
            "loss_cls": {
                "type": "QualityFocalLoss",
                "use_sigmoid": True,
                "beta": 2.0,
                "loss_weight": 1.0,
            },
            "loss_bbox": {"type": "GIoULoss", "loss_weight": 2.0},
            "loss_mask": {
                "type": "DiceLoss",
                "loss_weight": 2.0,
                "eps": 0.000005,
                "reduction": "mean",
            },
        },
        "train_cfg": {
            "assigner": {"type": "DynamicSoftLabelAssigner", "topk": assigner_topk},
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        },
        "test_cfg": {
            "nms_pre": max(3000, int(inference_postprocess["max_per_img"])),
            "min_bbox_size": 0,
            "score_thr": inference_postprocess["score_thr"],
            "nms": inference_postprocess["nms"],
            "max_per_img": inference_postprocess["max_per_img"],
            "mask_thr_binary": 0.5,
        },
    }


def _build_rtmdet_optim_wrapper(project_cfg: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = project_cfg["optimizer"]
    optim_wrapper: dict[str, Any] = {
        "type": "OptimWrapper",
        "optimizer": {
            "type": "AdamW",
            "lr": float(optimizer_cfg["lr"]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
        },
        "paramwise_cfg": {
            "norm_decay_mult": 0.0,
            "bias_decay_mult": 0.0,
            "bypass_duplicate": True,
        },
    }
    accumulative_counts = int(project_cfg["runtime"].get("accumulative_counts", 1))
    if accumulative_counts != 1:
        optim_wrapper["accumulative_counts"] = accumulative_counts
    return optim_wrapper


def build_mmdet_config_dict(experiment_config: str | Path) -> dict[str, Any]:
    config = build_mmdet_config_dict_for_mask_rcnn(
        experiment_config,
        build_model_config=build_model_config,
    )
    project_cfg = config["project_cfg"]
    model_size = _resolve_model_size(project_cfg)
    checkpoint = _resolve_pretrained_checkpoint(project_cfg, model_size)
    project_cfg["model"]["resolved_model_size"] = model_size
    project_cfg["model"]["resolved_pretrained_detector_checkpoint"] = checkpoint
    config["load_from"] = checkpoint
    config["optim_wrapper"] = _build_rtmdet_optim_wrapper(project_cfg)
    return config


__all__ = ["build_mmdet_config_dict", "build_model_config"]
