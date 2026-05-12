from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.common.io import write_json
from src.data.build_folds import materialize_dev_split_assets
from src.data.materialize_tiled_training_dataset import ensure_tiled_training_dataset
from src.models.mmdet_mask_rcnn_common import (
    load_project_experiment_config,
    resolve_experiment_config_path,
    resolve_repo_path,
)
from src.models.mmdet_registry import (
    build_mmdet_config_dict_for_experiment,
    dump_resolved_config_artifacts_for_experiment,
    resolve_mmdet_experiment_bundle,
    resolve_bridge_config_path,
)


def _check_mmdet_runtime(*, requires_mmpretrain: bool = False) -> tuple[bool, str]:
    try:
        import mmcv  # noqa: F401
        import mmdet  # noqa: F401
        import mmengine  # noqa: F401
        if requires_mmpretrain:
            import mmpretrain  # noqa: F401
    except ImportError as error:
        return False, str(error)
    return True, "ok"


def _prepare_runtime_cache_env() -> None:
    cache_root = Path("/tmp/hw3_mmdet_runtime_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg_cache"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch_home"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _run_config_only_smoke(experiment_config: Path) -> int:
    config_dict = build_mmdet_config_dict_for_experiment(experiment_config)
    artifacts = dump_resolved_config_artifacts_for_experiment(experiment_config)
    print(json.dumps(
        {
            "experiment_name": config_dict["experiment_name"],
            "work_dir": config_dict["work_dir"],
            "train_ann_file": config_dict["dataset_assets"]["train_ann_file"],
            "val_ann_file": config_dict["dataset_assets"]["val_ann_file"],
            "resolved_json": artifacts["json"],
            "resolved_python": artifacts["python"],
        },
        indent=2,
    ))
    return 0


def _prepare_train_runtime_assets(experiment_config: Path) -> None:
    project_cfg = load_project_experiment_config(experiment_config)
    dataset_cfg = project_cfg.get("dataset", {})
    train_tiled_cfg = dict(dataset_cfg.get("train_tiled", {}))
    use_all_train_for_training = bool(dataset_cfg.get("use_all_train_for_training", False))

    if bool(train_tiled_cfg.get("enabled", False)):
        print(
            "[train-launch] Preparing tiled train dataset cache "
            f"for {project_cfg['experiment']['name']} "
            f"(tile_size={train_tiled_cfg.get('tile_size', 768)}, "
            f"overlap={train_tiled_cfg.get('overlap', 0.25)}, "
            f"min_visible_fraction={train_tiled_cfg.get('min_visible_fraction', 0.0)}).",
            flush=True,
        )
        train_ann_file = resolve_repo_path(dataset_cfg["train_coco_json"])
        ensure_kwargs: dict[str, Any] = {}
        if train_tiled_cfg.get("output_root"):
            ensure_kwargs["output_root"] = resolve_repo_path(train_tiled_cfg["output_root"])
        elif train_tiled_cfg.get("dataset_name"):
            ensure_kwargs["dataset_name"] = str(train_tiled_cfg["dataset_name"])
        if train_tiled_cfg.get("dataset_label"):
            ensure_kwargs["dataset_label"] = str(train_tiled_cfg["dataset_label"])

        if use_all_train_for_training:
            ensure_kwargs.setdefault("dataset_label", "all_train")
        else:
            split_assets = materialize_dev_split_assets(
                full_train_json=resolve_repo_path(dataset_cfg["train_coco_json"]),
                folds_json=resolve_repo_path(dataset_cfg["folds_json"]),
                fold_index=int(dataset_cfg["dev_fold_index"]),
            )
            train_ann_file = Path(split_assets["train_ann_file"])
            ensure_kwargs["fold_index"] = int(dataset_cfg["dev_fold_index"])
        tiled_assets = ensure_tiled_training_dataset(
            train_ann_file=Path(train_ann_file),
            train_image_root=resolve_repo_path(dataset_cfg["train_image_root"]),
            tile_size=train_tiled_cfg.get("tile_size", 768),
            overlap=float(train_tiled_cfg.get("overlap", 0.25)),
            min_tile_coverage=float(train_tiled_cfg.get("min_tile_coverage", 0.0)),
            drop_empty_tiles=bool(train_tiled_cfg.get("drop_empty_tiles", True)),
            min_visible_fraction=float(train_tiled_cfg.get("min_visible_fraction", 0.0)),
            min_gt_area=float(train_tiled_cfg.get("min_gt_area", 1.0)),
            force_rebuild=bool(train_tiled_cfg.get("force_rebuild", False)),
            **ensure_kwargs,
        )
        print(
            "[train-launch] Tiled train dataset is ready: "
            f"{tiled_assets['train_ann_file']}",
            flush=True,
        )

    for extra_cfg in dataset_cfg.get("additional_tiled_datasets", []):
        extra = dict(extra_cfg)
        label = str(extra.get("dataset_label", extra.get("name", "additional_tiled")))
        print(
            "[train-launch] Preparing additional tiled dataset cache "
            f"{label} (tile_size={extra.get('tile_size', 768)}, "
            f"overlap={extra.get('overlap', 0.0)}).",
            flush=True,
        )
        extra_assets = ensure_tiled_training_dataset(
            fold_index=None,
            train_ann_file=resolve_repo_path(extra["source_coco"]),
            train_image_root=resolve_repo_path(extra["image_root"]),
            output_root=resolve_repo_path(extra["output_root"])
            if extra.get("output_root")
            else None,
            dataset_name=str(extra["dataset_name"]) if extra.get("dataset_name") else None,
            dataset_label=label,
            tile_size=extra.get("tile_size", 768),
            overlap=float(extra.get("overlap", 0.0)),
            min_tile_coverage=float(extra.get("min_tile_coverage", 0.0)),
            drop_empty_tiles=bool(extra.get("drop_empty_tiles", True)),
            min_visible_fraction=float(extra.get("min_visible_fraction", 0.5)),
            min_gt_area=float(extra.get("min_gt_area", 1.0)),
            force_rebuild=bool(extra.get("force_rebuild", False)),
        )
        print(
            "[train-launch] Additional tiled dataset is ready: "
            f"{extra_assets['train_ann_file']}",
            flush=True,
        )


def _apply_runtime_smoke_overrides(cfg: Any) -> None:
    """Keep the real MMDetection train path tiny and offline-safe."""
    smoke_work_dir = Path("outputs/runs/smoke_mmdet_runtime").resolve()
    cfg.work_dir = str(smoke_work_dir)
    cfg.load_from = None
    cfg.model.backbone.init_cfg = None

    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.dataset.indices = [0]

    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 999
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None

    cfg.default_hooks.logger.interval = 1
    cfg.default_hooks.checkpoint = {
        "type": "CheckpointHook",
        "interval": 1,
        "max_keep_ckpts": 1,
    }


def _apply_short_train_overrides(cfg: Any) -> None:
    """Run one real fold0 epoch without changing the baseline config lineage."""
    base_work_dir = Path(cfg.work_dir).resolve()
    cfg.work_dir = str(base_work_dir.with_name(f"{base_work_dir.name}_short"))

    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 2
    cfg.train_dataloader.persistent_workers = False
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = 2
    cfg.val_dataloader.persistent_workers = False

    cfg.optim_wrapper.optimizer.lr = 0.00125
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 1
    cfg.param_scheduler[1].end = 1
    cfg.param_scheduler[1].milestones = []

    cfg.default_hooks.logger.interval = 10
    cfg.default_hooks.checkpoint.interval = cfg.train_cfg.max_epochs
    cfg.default_hooks.checkpoint.max_keep_ckpts = 1
    cfg.default_hooks.checkpoint.save_last = True


def _apply_inference_overrides(
    cfg: Any,
    inference_overrides: dict[str, Any] | None = None,
    output_suffix: str | None = None,
) -> None:
    if inference_overrides:
        model_test_cfg = (
            cfg.model.test_cfg.rcnn
            if hasattr(cfg.model.test_cfg, "rcnn")
            else cfg.model.test_cfg
        )
        if "score_thr" in inference_overrides:
            model_test_cfg.score_thr = float(inference_overrides["score_thr"])
        if "max_per_img" in inference_overrides:
            model_test_cfg.max_per_img = int(inference_overrides["max_per_img"])
        if "nms_iou_threshold" in inference_overrides:
            model_test_cfg.nms.iou_threshold = float(inference_overrides["nms_iou_threshold"])

    if output_suffix:
        cfg.test_evaluator.outfile_prefix = f"{cfg.test_evaluator.outfile_prefix}_{output_suffix}"


def _apply_val_only_overrides(cfg: Any) -> None:
    """Disable train-time hooks that break standalone validation passes."""
    cfg.default_hooks.checkpoint = None


def _run_with_mmdet(
    action: str,
    experiment_config: Path,
    checkpoint: Path | None = None,
    inference_overrides: dict[str, Any] | None = None,
    output_suffix: str | None = None,
    metrics_output: Path | None = None,
) -> int | dict[str, Any]:
    _prepare_runtime_cache_env()

    from mmdet.utils import register_all_modules
    from mmengine.config import Config
    from mmengine.runner import Runner

    register_all_modules(init_default_scope=False)
    os.environ["HW3_EXPERIMENT_CONFIG"] = str(experiment_config)

    config_path = resolve_bridge_config_path(experiment_config)
    cfg = Config.fromfile(str(config_path), lazy_import=False)
    if checkpoint is not None:
        cfg.load_from = str(checkpoint.resolve())
    if action == "smoke-train":
        _apply_runtime_smoke_overrides(cfg)
    elif action == "short-train":
        _apply_short_train_overrides(cfg)
    elif action in {"test", "val"}:
        _apply_inference_overrides(
            cfg,
            inference_overrides=inference_overrides,
            output_suffix=output_suffix,
        )
        if action == "val":
            _apply_val_only_overrides(cfg)

    runner = Runner.from_cfg(cfg)
    if action in {"train", "smoke-train", "short-train"}:
        runner.train()
    elif action == "val":
        metrics = _json_safe(runner.val())
        if metrics_output is not None:
            write_json(metrics_output, metrics)
        return metrics
    elif action == "test":
        runner.test()
    else:
        raise ValueError(f"Unsupported MMDetection action: {action}")
    return 0


def _run_checkpoint_load_smoke(experiment_config: Path) -> int:
    _prepare_runtime_cache_env()

    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint

    def _collect_num_classes(head: Any) -> Any:
        if hasattr(head, "num_classes"):
            return int(head.num_classes)
        if isinstance(head, (list, tuple)):
            return [_collect_num_classes(item) for item in head]
        if hasattr(head, "__iter__"):
            try:
                return [_collect_num_classes(item) for item in head]
            except TypeError:
                pass
        return None

    def _summarize_model_heads(model: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {"model_type": type(model).__name__}
        if hasattr(model, "roi_head"):
            roi_head = model.roi_head
            summary["bbox_num_classes"] = _collect_num_classes(roi_head.bbox_head)
            summary["mask_num_classes"] = _collect_num_classes(roi_head.mask_head)
            return summary
        if hasattr(model, "bbox_head"):
            bbox_head = model.bbox_head
            summary["bbox_num_classes"] = _collect_num_classes(bbox_head)
            summary["mask_num_classes"] = _collect_num_classes(bbox_head)
            summary["head_type"] = type(bbox_head).__name__
            return summary
        summary["bbox_num_classes"] = None
        summary["mask_num_classes"] = None
        return summary

    register_all_modules(init_default_scope=False)
    init_default_scope("mmdet")
    os.environ["HW3_EXPERIMENT_CONFIG"] = str(experiment_config)

    config_path = resolve_bridge_config_path(experiment_config)
    cfg = Config.fromfile(str(config_path), lazy_import=False)
    load_from = getattr(cfg, "load_from", None)

    def _find_backbone_pretrained_init() -> dict[str, Any] | None:
        model_cfg = getattr(cfg, "model", {})
        backbone_cfg = model_cfg.get("backbone", {}) if hasattr(model_cfg, "get") else {}
        init_cfg = backbone_cfg.get("init_cfg") if hasattr(backbone_cfg, "get") else None
        init_items = init_cfg if isinstance(init_cfg, list) else [init_cfg]
        for item in init_items:
            if not item or not hasattr(item, "get"):
                continue
            if item.get("type") == "Pretrained" and item.get("checkpoint"):
                return {
                    "checkpoint": item["checkpoint"],
                    "prefix": item.get("prefix"),
                }
        return None

    backbone_pretrained = _find_backbone_pretrained_init()
    if not load_from and backbone_pretrained is None:
        raise ValueError(
            "No top-level `load_from` checkpoint or backbone pretrained `init_cfg` "
            f"is configured for {experiment_config}."
        )

    model = MODELS.build(cfg.model)
    smoke_summary = {
        "experiment_name": getattr(cfg, "hw3_experiment_name", experiment_config.stem),
    }
    if load_from:
        smoke_summary["pretrained_source"] = "load_from"
        smoke_summary["load_from"] = load_from
    else:
        smoke_summary["pretrained_source"] = "model.backbone.init_cfg"
        smoke_summary["backbone_init_checkpoint"] = backbone_pretrained["checkpoint"]
        smoke_summary["backbone_init_prefix"] = backbone_pretrained["prefix"]
    smoke_summary.update(_summarize_model_heads(model))
    print(json.dumps(smoke_summary, indent=2))
    if load_from:
        load_checkpoint(model, load_from, map_location="cpu", strict=False, logger="current")
    else:
        # Backbone-only ImageNet initialization is valid for ConvNeXt-style experiments.
        # Calling the backbone init path keeps this smoke lightweight and avoids
        # pretending there is a full detector checkpoint to load.
        model.backbone.init_weights()
    return 0


def run_mmdet_action(
    action: str,
    experiment_config: Path,
    checkpoint: Path | None = None,
    inference_overrides: dict[str, Any] | None = None,
    output_suffix: str | None = None,
    metrics_output: Path | None = None,
) -> int | dict[str, Any]:
    resolved_experiment_config = resolve_experiment_config_path(experiment_config)
    experiment_bundle = resolve_mmdet_experiment_bundle(resolved_experiment_config)
    if action == "check":
        return _run_config_only_smoke(resolved_experiment_config)

    available, message = _check_mmdet_runtime(
        requires_mmpretrain=bool(experiment_bundle["requires_mmpretrain"])
    )
    if not available:
        print(
            "MMDetection runtime is not installed in the current environment. "
            "The repo wiring and config assembly are ready, but a real "
            f"`{action}` run is blocked until the required OpenMMLab packages are installed."
        )
        print(f"Import failure: {message}")
        return 1
    if action == "checkpoint-load-smoke":
        return _run_checkpoint_load_smoke(resolved_experiment_config)
    if action in {"train", "smoke-train", "short-train"}:
        _prepare_train_runtime_assets(resolved_experiment_config)
    return _run_with_mmdet(
        action,
        resolved_experiment_config,
        checkpoint=checkpoint,
        inference_overrides=inference_overrides,
        output_suffix=output_suffix,
        metrics_output=metrics_output,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Thin MMDetection launcher wrapper.")
    parser.add_argument(
        "--action",
        choices=[
            "check",
            "checkpoint-load-smoke",
            "smoke-train",
            "short-train",
            "train",
            "val",
            "test",
        ],
        default="check",
        help="Run a config-only smoke check or a real MMDetection action.",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/exp001_mmdet_maskrcnn_r50.yaml"),
        help="Experiment-facing config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load for test/predict flows.",
    )
    args = parser.parse_args()
    result = run_mmdet_action(args.action, args.experiment, checkpoint=args.checkpoint)
    if isinstance(result, int):
        return result
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
