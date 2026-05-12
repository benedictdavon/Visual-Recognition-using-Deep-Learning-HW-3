from __future__ import annotations

import json
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

from src.common.config import CONFIG_ARTIFACTS_DIR, PROJECT_ROOT, RUNS_DIR
from src.common.io import write_json, write_text
from src.common.config_loader import load_config_with_bases
from src.data.materialize_tiled_training_dataset import resolve_tiled_training_dataset_assets
from src.data.register_mmdet_dataset import build_mmdet_dataset_bundle, build_mmdet_dataset_dict


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_experiment_config_path(
    path: str | Path | None = None,
    *,
    default_experiment_config: str | Path | None = None,
) -> Path:
    configured = path or os.environ.get("HW3_EXPERIMENT_CONFIG") or default_experiment_config
    if configured is None:
        raise ValueError("No experiment config path was provided and no default is available.")
    return resolve_repo_path(configured)


def _as_scale_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected MMDetection scale as length-2 list/tuple, got: {value!r}")


def _as_spatial_size_tuple(value: Any, *, field_name: str) -> tuple[int, int]:
    if isinstance(value, int):
        size = int(value)
        if size <= 0:
            raise ValueError(f"{field_name} must be positive, got {value!r}")
        return (size, size)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height = int(value[0])
        width = int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"{field_name} values must be positive, got {value!r}")
        return (height, width)
    raise ValueError(
        f"Expected {field_name} as a positive integer or length-2 list/tuple, got: {value!r}"
    )


def _build_resize_transform(scale_config: Any) -> dict[str, Any]:
    if (
        isinstance(scale_config, (list, tuple))
        and scale_config
        and isinstance(scale_config[0], (list, tuple))
    ):
        return {
            "type": "RandomChoiceResize",
            "scales": [_as_scale_tuple(scale) for scale in scale_config],
            "keep_ratio": True,
        }
    return {"type": "Resize", "scale": _as_scale_tuple(scale_config), "keep_ratio": True}


def _validate_probability(name: str, value: Any, *, allow_zero: bool = True) -> float:
    numeric = float(value)
    lower_bound = 0.0 if allow_zero else 0.0
    if numeric < lower_bound or numeric > 1.0 or (not allow_zero and numeric == 0.0):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be within {interval}, got {value!r}")
    return numeric


def _build_train_crop_transforms(project_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    crop_cfg = project_cfg.get("augmentation", {}).get("train", {}).get("crop")
    if not crop_cfg or not bool(crop_cfg.get("enabled", False)):
        return []

    transforms: list[dict[str, Any]] = [
        {
            "type": "RandomCrop",
            "crop_type": str(crop_cfg.get("crop_type", "absolute")),
            "crop_size": _as_spatial_size_tuple(
                crop_cfg.get("size", crop_cfg.get("crop_size", 768)),
                field_name="augmentation.train.crop.size",
            ),
            "allow_negative_crop": bool(crop_cfg.get("allow_negative_crop", False)),
            "bbox_clip_border": bool(crop_cfg.get("bbox_clip_border", True)),
            "recompute_bbox": bool(crop_cfg.get("recompute_bbox", True)),
        }
    ]

    filter_cfg = crop_cfg.get("filter_annotations", {})
    if filter_cfg is None:
        filter_cfg = {}
    if bool(filter_cfg.get("enabled", True)):
        transforms.append(
            {
                "type": "FilterAnnotations",
                "min_gt_bbox_wh": [
                    float(value)
                    for value in filter_cfg.get("min_gt_bbox_wh", [0.01, 0.01])
                ],
                "by_mask": bool(filter_cfg.get("by_mask", True)),
                "keep_empty": bool(filter_cfg.get("keep_empty", False)),
            }
        )
    return transforms


def _resolve_inference_postprocess(project_cfg: dict[str, Any]) -> dict[str, Any]:
    inference_cfg = project_cfg.get("inference", {})
    postprocess_cfg = inference_cfg.get("postprocess", {})
    nms_cfg = dict(postprocess_cfg.get("nms", {"type": "nms", "iou_threshold": 0.5}))
    nms_iou = _validate_probability(
        "inference.postprocess.nms.iou_threshold",
        nms_cfg["iou_threshold"],
    )
    max_per_img = int(postprocess_cfg.get("max_per_img", 100))
    if max_per_img <= 0:
        raise ValueError(
            f"inference.postprocess.max_per_img must be positive, got {max_per_img!r}"
        )
    return {
        "score_thr": _validate_probability(
            "inference.postprocess.score_thr",
            postprocess_cfg.get("score_thr", 0.05),
        ),
        "max_per_img": max_per_img,
        "nms": {
            "type": nms_cfg.get("type", "nms"),
            "iou_threshold": nms_iou,
        },
    }


def load_project_experiment_config(experiment_config: str | Path) -> dict[str, Any]:
    resolved_experiment_config = resolve_repo_path(experiment_config)
    payload = load_config_with_bases(resolved_experiment_config)
    payload.setdefault("experiment", {})
    payload["experiment"].setdefault("config_path", str(resolved_experiment_config))
    payload["experiment"].setdefault(
        "name", resolved_experiment_config.stem.replace(".yaml", "")
    )
    payload.setdefault("dataset", {})
    payload.setdefault("runtime", {})
    payload.setdefault("augmentation", {})
    payload.setdefault("optimizer", {})
    payload.setdefault("schedule", {})
    payload.setdefault("model", {})
    payload.setdefault("inference", {})
    payload["inference"].setdefault("postprocess", {})
    payload["inference"].setdefault("sweep", {})
    payload["runtime"].setdefault(
        "work_dir",
        str((RUNS_DIR / payload["experiment"]["name"]).resolve()),
    )
    return payload


def build_train_pipeline(project_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    train_aug = project_cfg["augmentation"]["train"]
    crop_cfg = train_aug.get("crop", {})
    crop_transforms = _build_train_crop_transforms(project_cfg)
    disable_resize = bool(train_aug.get("disable_resize", False))
    pipeline = [
        {"type": "LoadImageFromFile", "color_type": project_cfg["dataset"]["image_color_type"]},
        {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
    ]
    pipeline.extend(crop_transforms)
    if (not disable_resize) and (
        (not crop_transforms) or bool(crop_cfg.get("resize_after_crop", False))
    ):
        pipeline.append(_build_resize_transform(train_aug["scale"]))
    pipeline.extend(
        [
            {"type": "RandomFlip", "prob": train_aug["random_flip_prob"]},
            {"type": "PackDetInputs"},
        ]
    )
    return pipeline


def build_test_pipeline(project_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    test_aug = project_cfg["augmentation"]["test"]
    return [
        {"type": "LoadImageFromFile", "color_type": project_cfg["dataset"]["image_color_type"]},
        {"type": "Resize", "scale": _as_scale_tuple(test_aug["scale"]), "keep_ratio": True},
        {
            "type": "PackDetInputs",
            "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        },
    ]


def build_val_pipeline(project_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    val_pipeline = build_test_pipeline(project_cfg)
    val_pipeline.insert(2, {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True})
    return val_pipeline


def build_train_dataset_config(
    project_cfg: dict[str, Any],
    train_dataset: dict[str, Any],
) -> dict[str, Any]:
    wrapper_cfg = project_cfg.get("dataset", {}).get("train_wrapper")
    if not wrapper_cfg:
        return train_dataset

    wrapped_dataset = dict(train_dataset)
    wrapper_type = str(wrapper_cfg.get("type", "")).strip()
    if not wrapper_type:
        raise ValueError("dataset.train_wrapper.type must be set when dataset.train_wrapper is used.")
    if "dataset" in wrapper_cfg:
        raise ValueError(
            "dataset.train_wrapper should not set its own nested dataset; "
            "the shared MMDetection builder injects the base train dataset."
        )

    wrapper = dict(wrapper_cfg)
    if wrapper_type == "ClassBalancedDataset":
        oversample_thr = float(wrapper.get("oversample_thr", 0.0))
        if oversample_thr <= 0.0 or oversample_thr > 1.0:
            raise ValueError(
                "dataset.train_wrapper.oversample_thr must be within (0, 1], "
                f"got {oversample_thr!r}."
            )
        wrapper["oversample_thr"] = oversample_thr

    wrapper["type"] = wrapper_type
    wrapper["dataset"] = wrapped_dataset
    return wrapper


def _build_concat_train_dataset(
    *,
    full_train_dataset: dict[str, Any],
    tiled_train_dataset: dict[str, Any],
    train_mix_cfg: dict[str, Any],
) -> dict[str, Any]:
    dataset_type = str(train_mix_cfg.get("type", "ConcatDataset")).strip() or "ConcatDataset"
    if dataset_type != "ConcatDataset":
        raise ValueError(
            "dataset.train_mix.type currently supports only 'ConcatDataset', "
            f"got {dataset_type!r}."
        )
    return {
        "type": dataset_type,
        "datasets": [dict(full_train_dataset), dict(tiled_train_dataset)],
    }


def _resolve_additional_tiled_dataset_assets(
    dataset_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for index, extra_cfg in enumerate(dataset_cfg.get("additional_tiled_datasets", [])):
        extra = dict(extra_cfg)
        if "source_coco" not in extra:
            raise ValueError(
                f"dataset.additional_tiled_datasets[{index}] is missing `source_coco`."
            )
        if "image_root" not in extra:
            raise ValueError(
                f"dataset.additional_tiled_datasets[{index}] is missing `image_root`."
            )
        if not extra.get("output_root") and not extra.get("dataset_name"):
            raise ValueError(
                "Additional tiled datasets must set either `output_root` or `dataset_name` "
                f"(item index {index})."
            )

        asset_kwargs: dict[str, Any] = {
            "tile_size": extra.get("tile_size", 768),
            "overlap": float(extra.get("overlap", 0.0)),
        }
        if extra.get("output_root"):
            asset_kwargs["output_root"] = resolve_repo_path(extra["output_root"])
        else:
            asset_kwargs["dataset_name"] = str(extra["dataset_name"])
        tiled_assets = resolve_tiled_training_dataset_assets(**asset_kwargs)
        assets.append(
            {
                "name": str(extra.get("name", extra.get("dataset_label", f"additional_{index}"))),
                "source_coco": str(resolve_repo_path(extra["source_coco"])),
                "image_root": str(resolve_repo_path(extra["image_root"])),
                "train_ann_file": str(resolve_repo_path(tiled_assets["train_ann_file"])),
                "train_image_root": str(resolve_repo_path(tiled_assets["train_image_root"])),
                "summary_file": str(resolve_repo_path(tiled_assets["summary_file"])),
            }
        )
    return assets


def _build_checkpoint_hook_config(
    runtime_cfg: dict[str, Any],
    *,
    max_epochs: int,
    disable_validation: bool,
) -> dict[str, Any]:
    checkpoint_cfg = dict(runtime_cfg.get("checkpoint", {}))
    hook = {
        "type": "CheckpointHook",
        "interval": int(checkpoint_cfg.get("interval", max_epochs)),
        "by_epoch": bool(checkpoint_cfg.get("by_epoch", True)),
        "save_last": bool(checkpoint_cfg.get("save_last", True)),
        "max_keep_ckpts": int(checkpoint_cfg.get("max_keep_ckpts", 1)),
    }
    if hook["interval"] <= 0:
        raise ValueError(
            "runtime.checkpoint.interval must be positive, "
            f"got {hook['interval']!r}."
        )
    if hook["max_keep_ckpts"] == 0 or hook["max_keep_ckpts"] < -1:
        raise ValueError(
            "runtime.checkpoint.max_keep_ckpts must be positive or -1, "
            f"got {hook['max_keep_ckpts']!r}."
        )

    if "save_begin" in checkpoint_cfg:
        save_begin = int(checkpoint_cfg["save_begin"])
        if save_begin < 0:
            raise ValueError(
                "runtime.checkpoint.save_begin must be non-negative, "
                f"got {save_begin!r}."
            )
        hook["save_begin"] = save_begin

    if "filename_tmpl" in checkpoint_cfg:
        hook["filename_tmpl"] = str(checkpoint_cfg["filename_tmpl"])

    explicit_save_best = checkpoint_cfg.get("save_best", "__hw3_default__")
    if explicit_save_best == "__hw3_default__":
        if not disable_validation:
            hook["save_best"] = "coco/segm_mAP_50"
            hook["rule"] = "greater"
    elif explicit_save_best is not None:
        hook["save_best"] = explicit_save_best
        if checkpoint_cfg.get("rule") is not None:
            hook["rule"] = str(checkpoint_cfg["rule"])
    return hook


def build_mask_rcnn_model_config(
    project_cfg: dict[str, Any],
    *,
    backbone: dict[str, Any],
    neck: dict[str, Any],
) -> dict[str, Any]:
    num_classes = int(project_cfg["model"]["num_classes"])
    inference_postprocess = _resolve_inference_postprocess(project_cfg)
    return {
        "type": "MaskRCNN",
        "data_preprocessor": {
            "type": "DetDataPreprocessor",
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "bgr_to_rgb": True,
            "pad_mask": True,
            "pad_size_divisor": 32,
        },
        "backbone": backbone,
        "neck": neck,
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
            "loss_bbox": {"type": "L1Loss", "loss_weight": 1.0},
        },
        "roi_head": {
            "type": "StandardRoIHead",
            "bbox_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {"type": "RoIAlign", "output_size": 7, "sampling_ratio": 0},
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32],
            },
            "bbox_head": {
                "type": "Shared2FCBBoxHead",
                "in_channels": 256,
                "fc_out_channels": 1024,
                "roi_feat_size": 7,
                "num_classes": num_classes,
                "bbox_coder": {
                    "type": "DeltaXYWHBBoxCoder",
                    "target_means": [0.0, 0.0, 0.0, 0.0],
                    "target_stds": [0.1, 0.1, 0.2, 0.2],
                },
                "reg_class_agnostic": False,
                "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 1.0},
                "loss_bbox": {"type": "L1Loss", "loss_weight": 1.0},
            },
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
        "train_cfg": {
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
                "allowed_border": -1,
                "pos_weight": -1,
                "debug": False,
            },
            "rpn_proposal": {
                "nms_pre": 2000,
                "max_per_img": 1000,
                "nms": {"type": "nms", "iou_threshold": 0.7},
                "min_bbox_size": 0,
            },
            "rcnn": {
                "assigner": {
                    "type": "MaxIoUAssigner",
                    "pos_iou_thr": 0.5,
                    "neg_iou_thr": 0.5,
                    "min_pos_iou": 0.5,
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
            },
        },
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


def build_mmdet_config_dict_for_mask_rcnn(
    experiment_config: str | Path,
    *,
    build_model_config: Callable[[dict[str, Any]], dict[str, Any]],
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_cfg = load_project_experiment_config(experiment_config)
    dataset_cfg = project_cfg["dataset"]
    runtime_cfg = project_cfg["runtime"]
    use_all_train_for_training = bool(dataset_cfg.get("use_all_train_for_training", False))
    disable_validation = bool(runtime_cfg.get("disable_validation", False))
    train_pipeline = build_train_pipeline(project_cfg)
    val_pipeline = build_val_pipeline(project_cfg)
    test_pipeline = build_test_pipeline(project_cfg)
    dataset_bundle = build_mmdet_dataset_bundle(
        dev_fold_index=int(dataset_cfg["dev_fold_index"]),
        full_train_json=resolve_repo_path(dataset_cfg["train_coco_json"]),
        test_image_info_json=resolve_repo_path(dataset_cfg["test_image_info_json"]),
        folds_json=resolve_repo_path(dataset_cfg["folds_json"]),
        train_image_root=resolve_repo_path(dataset_cfg["train_image_root"]),
        test_image_root=resolve_repo_path(dataset_cfg["test_image_root"]),
        train_pipeline=train_pipeline,
        val_pipeline=val_pipeline,
        test_pipeline=test_pipeline,
    )
    base_train_dataset = (
        build_mmdet_dataset_dict(
            ann_file=resolve_repo_path(dataset_cfg["train_coco_json"]),
            image_root=resolve_repo_path(dataset_cfg["train_image_root"]),
            pipeline=train_pipeline,
            test_mode=False,
        )
        if use_all_train_for_training
        else dict(dataset_bundle["train_dataset"])
    )
    train_dataset = dict(base_train_dataset)
    train_tiled_cfg = dict(dataset_cfg.get("train_tiled", {}))
    train_mix_cfg = dict(dataset_cfg.get("train_mix", {}))
    tiled_train_assets: dict[str, str] | None = None
    tiled_train_dataset: dict[str, Any] | None = None
    additional_tiled_assets = _resolve_additional_tiled_dataset_assets(dataset_cfg)
    additional_tiled_datasets: list[dict[str, Any]] = []
    if bool(train_tiled_cfg.get("enabled", False)):
        tiled_asset_kwargs: dict[str, Any] = {
            "tile_size": train_tiled_cfg.get("tile_size", 768),
            "overlap": float(train_tiled_cfg.get("overlap", 0.25)),
        }
        if train_tiled_cfg.get("output_root"):
            tiled_asset_kwargs["output_root"] = resolve_repo_path(train_tiled_cfg["output_root"])
        elif train_tiled_cfg.get("dataset_name"):
            tiled_asset_kwargs["dataset_name"] = str(train_tiled_cfg["dataset_name"])
        else:
            tiled_asset_kwargs["fold_index"] = (
                None if use_all_train_for_training else int(dataset_cfg["dev_fold_index"])
            )
        tiled_train_assets = resolve_tiled_training_dataset_assets(**tiled_asset_kwargs)
        tiled_train_dataset = build_mmdet_dataset_dict(
            ann_file=resolve_repo_path(tiled_train_assets["train_ann_file"]),
            image_root=resolve_repo_path(tiled_train_assets["train_image_root"]),
            pipeline=train_pipeline,
            test_mode=False,
        )
        if bool(train_mix_cfg.get("enabled", False)):
            train_dataset = _build_concat_train_dataset(
                full_train_dataset=base_train_dataset,
                tiled_train_dataset=tiled_train_dataset,
                train_mix_cfg=train_mix_cfg,
            )
        else:
            train_dataset = tiled_train_dataset

    for extra_assets in additional_tiled_assets:
        additional_tiled_datasets.append(
            build_mmdet_dataset_dict(
                ann_file=resolve_repo_path(extra_assets["train_ann_file"]),
                image_root=resolve_repo_path(extra_assets["train_image_root"]),
                pipeline=train_pipeline,
                test_mode=False,
            )
        )
    if additional_tiled_datasets:
        train_dataset = {
            "type": "ConcatDataset",
            "datasets": [dict(train_dataset), *additional_tiled_datasets],
        }

    optimizer_cfg = project_cfg["optimizer"]
    schedule_cfg = project_cfg["schedule"]
    work_dir = runtime_cfg["work_dir"]
    test_outfile_prefix = str((resolve_repo_path(work_dir) / "test_release_predictions").resolve())
    max_epochs = int(runtime_cfg["max_epochs"])
    train_dataset_config = build_train_dataset_config(project_cfg, train_dataset)
    checkpoint_hook = _build_checkpoint_hook_config(
        runtime_cfg,
        max_epochs=max_epochs,
        disable_validation=disable_validation,
    )

    optim_wrapper = {
        "type": "OptimWrapper",
        "optimizer": {
            "type": "SGD",
            "lr": float(optimizer_cfg["lr"]),
            "momentum": float(optimizer_cfg["momentum"]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
        },
    }
    accumulative_counts = int(runtime_cfg.get("accumulative_counts", 1))
    if accumulative_counts <= 0:
        raise ValueError(
            "runtime.accumulative_counts must be positive, "
            f"got {accumulative_counts!r}."
        )
    if accumulative_counts != 1:
        optim_wrapper["accumulative_counts"] = accumulative_counts

    config = {
        "default_scope": "mmdet",
        "experiment_name": project_cfg["experiment"]["name"],
        "project_cfg": project_cfg,
        "dataset_assets": dataset_bundle["split_assets"],
        "metainfo": dataset_bundle["metainfo"],
        "model": build_model_config(project_cfg),
        "train_dataloader": {
            "batch_size": int(runtime_cfg["train_batch_size"]),
            "num_workers": int(runtime_cfg["num_workers"]),
            "persistent_workers": bool(runtime_cfg["persistent_workers"]),
            "sampler": {"type": "DefaultSampler", "shuffle": True},
            "batch_sampler": {"type": "AspectRatioBatchSampler"},
            "dataset": train_dataset_config,
        },
        "val_dataloader": (
            {
                "batch_size": int(runtime_cfg["val_batch_size"]),
                "num_workers": int(runtime_cfg["num_workers"]),
                "persistent_workers": bool(runtime_cfg["persistent_workers"]),
                "drop_last": False,
                "sampler": {"type": "DefaultSampler", "shuffle": False},
                "dataset": dataset_bundle["val_dataset"],
            }
            if not disable_validation
            else None
        ),
        "test_dataloader": {
            "batch_size": int(runtime_cfg["val_batch_size"]),
            "num_workers": int(runtime_cfg["num_workers"]),
            "persistent_workers": bool(runtime_cfg["persistent_workers"]),
            "drop_last": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": dataset_bundle["test_dataset"],
        },
        "val_evaluator": (
            {
                "type": "CocoMetric",
                "ann_file": dataset_bundle["split_assets"]["val_ann_file"],
                "metric": ["bbox", "segm"],
                "format_only": False,
            }
            if not disable_validation
            else None
        ),
        "test_evaluator": {
            "type": "CocoMetric",
            "ann_file": str(resolve_repo_path(project_cfg["dataset"]["test_image_info_json"])),
            "metric": ["bbox", "segm"],
            "format_only": True,
            "outfile_prefix": test_outfile_prefix,
        },
        "train_cfg": {
            "type": "EpochBasedTrainLoop",
            "max_epochs": max_epochs,
            "val_interval": (
                int(runtime_cfg["val_interval"])
                if not disable_validation
                else max_epochs + 1
            ),
        },
        "val_cfg": {"type": "ValLoop"} if not disable_validation else None,
        "test_cfg": {"type": "TestLoop"},
        "optim_wrapper": optim_wrapper,
        "param_scheduler": [
            {
                "type": "LinearLR",
                "start_factor": float(schedule_cfg["warmup_start_factor"]),
                "by_epoch": False,
                "begin": 0,
                "end": int(schedule_cfg["warmup_iters"]),
            },
            {
                "type": "MultiStepLR",
                "begin": 0,
                "end": max_epochs,
                "by_epoch": True,
                "milestones": [int(step) for step in schedule_cfg["milestones"]],
                "gamma": float(schedule_cfg["gamma"]),
            },
        ],
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": int(runtime_cfg["log_interval"])},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": checkpoint_hook,
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
        "env_cfg": {
            "cudnn_benchmark": False,
            "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
            "dist_cfg": {"backend": "nccl"},
        },
        "visualizer": {
            "type": "DetLocalVisualizer",
            "vis_backends": [{"type": "LocalVisBackend"}],
            "name": "visualizer",
        },
        "log_processor": {"type": "LogProcessor", "window_size": 50, "by_epoch": True},
        "log_level": "INFO",
        "resume": False,
        "work_dir": work_dir,
        "seed": int(runtime_cfg["seed"]),
        "launcher": runtime_cfg["launcher"],
        "auto_scale_lr": {
            "enable": False,
            "base_batch_size": int(runtime_cfg["auto_scale_base_batch_size"]),
        },
    }
    if tiled_train_assets is not None:
        config["dataset_assets"].update(
            {
                "tiled_train_ann_file": str(
                    resolve_repo_path(tiled_train_assets["train_ann_file"])
                ),
                "tiled_train_image_root": str(
                    resolve_repo_path(tiled_train_assets["train_image_root"])
                ),
                "tiled_train_summary_file": str(
                    resolve_repo_path(tiled_train_assets["summary_file"])
                ),
            }
        )
    if additional_tiled_assets:
        config["dataset_assets"]["additional_tiled_datasets"] = additional_tiled_assets
    if use_all_train_for_training:
        config["dataset_assets"].update(
            {
                "all_train_ann_file": str(resolve_repo_path(dataset_cfg["train_coco_json"])),
                "all_train_image_root": str(resolve_repo_path(dataset_cfg["train_image_root"])),
            }
        )
    if "custom_imports" in project_cfg:
        config["custom_imports"] = project_cfg["custom_imports"]
    if "custom_hooks" in project_cfg:
        config["custom_hooks"] = project_cfg["custom_hooks"]
    if extra_config:
        config.update(extra_config)
    return config


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def build_mmengine_config_globals(
    path: str | Path | None,
    *,
    default_experiment_config: str | Path,
    build_mmdet_config_dict: Callable[[str | Path], dict[str, Any]],
) -> dict[str, Any]:
    config_dict = build_mmdet_config_dict(
        resolve_experiment_config_path(
            path,
            default_experiment_config=default_experiment_config,
        )
    )
    globals_dict = {
        key: value
        for key, value in config_dict.items()
        if key not in {"project_cfg", "dataset_assets", "experiment_name"}
    }
    globals_dict["hw3_project_cfg"] = config_dict["project_cfg"]
    globals_dict["hw3_dataset_assets"] = config_dict["dataset_assets"]
    globals_dict["hw3_experiment_name"] = config_dict["experiment_name"]
    return globals_dict


def dump_resolved_config_artifacts(
    path: str | Path | None,
    *,
    default_experiment_config: str | Path,
    build_mmdet_config_dict: Callable[[str | Path], dict[str, Any]],
) -> dict[str, str]:
    experiment_path = resolve_experiment_config_path(
        path,
        default_experiment_config=default_experiment_config,
    )
    config_dict = build_mmdet_config_dict(experiment_path)
    safe_config_dict = _json_safe(config_dict)
    CONFIG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = config_dict["experiment_name"]
    json_path = CONFIG_ARTIFACTS_DIR / f"{prefix}_resolved.json"
    py_path = CONFIG_ARTIFACTS_DIR / f"{prefix}_resolved.py"

    write_json(json_path, safe_config_dict)
    write_text(py_path, "config = " + pformat(safe_config_dict, width=100) + "\n")
    return {"json": str(json_path), "python": str(py_path)}


def resolve_project_path(value: str | Path) -> Path:
    return resolve_repo_path(value)


def resolve_inference_postprocess(project_cfg: dict[str, Any]) -> dict[str, Any]:
    return _resolve_inference_postprocess(project_cfg)


def build_mask_rcnn_model(
    *,
    backbone: dict[str, Any],
    neck: dict[str, Any],
    num_classes: int,
    inference_postprocess: dict[str, Any],
) -> dict[str, Any]:
    project_cfg = {
        "model": {"num_classes": num_classes},
        "inference": {"postprocess": inference_postprocess},
    }
    return build_mask_rcnn_model_config(
        project_cfg,
        backbone=backbone,
        neck=neck,
    )


def build_mask_rcnn_experiment_config(
    experiment_config: str | Path,
    build_model_config: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    extra_top_level: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_mmdet_config_dict_for_mask_rcnn(
        experiment_config,
        build_model_config=build_model_config,
        extra_config=extra_top_level,
    )
