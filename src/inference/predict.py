from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from src.common.io import write_json
from src.inference.tile_inference import (
    append_tile_merge_output_suffix,
    build_tile_output_suffix,
    resolve_tile_config,
    resolve_tile_merge_postprocess_config,
    run_tiled_inference,
)
from src.models.mmdet_mask_rcnn_common import load_project_experiment_config
from src.training.launch_mmdet import run_mmdet_action


def _dedupe_preserve_order(values: list[Any]) -> list[Any]:
    ordered: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _validate_probability(name: str, value: float) -> float:
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value!r}")
    return numeric


def _validate_positive_int(name: str, value: int) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return numeric


def _format_float_token(value: float) -> str:
    return format(float(value), ".4f").rstrip("0").rstrip(".").replace(".", "p")


def _build_output_suffix(score_thr: float, max_per_img: int, nms_iou_threshold: float) -> str:
    return (
        f"score{_format_float_token(score_thr)}"
        f"_max{int(max_per_img)}"
        f"_nms{_format_float_token(nms_iou_threshold)}"
    )


def _default_sweep_dir(experiment_config: Path) -> Path:
    project_cfg = load_project_experiment_config(experiment_config)
    return Path(project_cfg["runtime"]["work_dir"]).resolve() / "inference_sweeps"


def _resolve_sweep_runs(
    experiment_config: Path,
    use_config_sweep: bool,
    score_thr_values: list[float] | None,
    max_per_img_values: list[int] | None,
    nms_iou_threshold_values: list[float] | None,
    max_sweep_runs: int | None,
) -> list[dict[str, Any]]:
    project_cfg = load_project_experiment_config(experiment_config)
    postprocess_cfg = project_cfg["inference"]["postprocess"]
    sweep_cfg = project_cfg["inference"]["sweep"]

    base_score_thr = _validate_probability("inference.postprocess.score_thr", postprocess_cfg.get("score_thr", 0.05))
    base_max_per_img = _validate_positive_int(
        "inference.postprocess.max_per_img",
        postprocess_cfg.get("max_per_img", 100),
    )
    base_nms_iou = _validate_probability(
        "inference.postprocess.nms.iou_threshold",
        postprocess_cfg.get("nms", {}).get("iou_threshold", 0.5),
    )

    score_values = score_thr_values
    max_values = max_per_img_values
    nms_values = nms_iou_threshold_values

    if use_config_sweep:
        score_values = score_values or sweep_cfg.get("score_thr_values") or []
        max_values = max_values or sweep_cfg.get("max_per_img_values") or []
        nms_values = nms_values or sweep_cfg.get("nms_iou_threshold_values") or []

    resolved_score_values = _dedupe_preserve_order(
        [_validate_probability("score_thr", value) for value in (score_values or [base_score_thr])]
    )
    resolved_max_values = _dedupe_preserve_order(
        [_validate_positive_int("max_per_img", value) for value in (max_values or [base_max_per_img])]
    )
    resolved_nms_values = _dedupe_preserve_order(
        [_validate_probability("nms_iou_threshold", value) for value in (nms_values or [base_nms_iou])]
    )

    sweep_runs = [
        {
            "score_thr": score_thr,
            "max_per_img": max_per_img,
            "nms_iou_threshold": nms_iou_threshold,
        }
        for score_thr, max_per_img, nms_iou_threshold in product(
            resolved_score_values,
            resolved_max_values,
            resolved_nms_values,
        )
    ]

    run_cap = int(max_sweep_runs if max_sweep_runs is not None else sweep_cfg.get("max_runs", 12))
    if run_cap <= 0:
        raise ValueError(f"max_sweep_runs must be positive, got {run_cap!r}")
    if len(sweep_runs) > run_cap:
        raise ValueError(
            f"Requested {len(sweep_runs)} inference runs, which exceeds the configured cap of {run_cap}. "
            "Reduce the sweep lists or raise the cap intentionally."
        )
    return sweep_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation/test inference through MMDetection.")
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
        help="Checkpoint to use for inference.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Evaluate on the fold0 validation split or export predictions on test_release.",
    )
    parser.add_argument(
        "--use-config-sweep",
        action="store_true",
        help="Use the optional inference.sweep values from the experiment config.",
    )
    parser.add_argument(
        "--score-thr",
        dest="score_thr_values",
        action="append",
        type=float,
        default=None,
        help="Repeat to sweep score thresholds, for example `--score-thr 0.03 --score-thr 0.05`.",
    )
    parser.add_argument(
        "--max-per-img",
        dest="max_per_img_values",
        action="append",
        type=int,
        default=None,
        help="Repeat to sweep max detections per image.",
    )
    parser.add_argument(
        "--nms-iou-thr",
        dest="nms_iou_threshold_values",
        action="append",
        type=float,
        default=None,
        help="Repeat to sweep NMS IoU thresholds.",
    )
    parser.add_argument(
        "--max-sweep-runs",
        type=int,
        default=None,
        help="Optional override for the configured sweep run cap.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for a validation sweep summary JSON.",
    )
    mask_score_group = parser.add_mutually_exclusive_group()
    mask_score_group.add_argument(
        "--mask-score-refine-enabled",
        dest="mask_score_refine_enabled",
        action="store_true",
        help="Enable bbox-score refinement using mask quality when soft masks are available.",
    )
    mask_score_group.add_argument(
        "--mask-score-refine-disabled",
        dest="mask_score_refine_enabled",
        action="store_false",
        help="Disable mask-score refinement even if the experiment config enables it.",
    )
    parser.set_defaults(mask_score_refine_enabled=None)
    parser.add_argument(
        "--mask-score-refine-alpha",
        type=float,
        default=None,
        help="Mask-score refinement exponent alpha in bbox_score * (mask_quality ** alpha).",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=None,
        help="Optional minimum merged mask area filter applied after tile merge.",
    )
    tile_mode_group = parser.add_mutually_exclusive_group()
    tile_mode_group.add_argument(
        "--tile-enabled",
        dest="tile_enabled",
        action="store_true",
        help="Enable sliding-window tiled inference.",
    )
    tile_mode_group.add_argument(
        "--tile-disabled",
        dest="tile_enabled",
        action="store_false",
        help="Disable tiled inference even if the experiment config enables it.",
    )
    parser.set_defaults(tile_enabled=None)
    parser.add_argument(
        "--tile-size",
        dest="tile_size",
        nargs="+",
        type=int,
        default=None,
        help="Tile size as one integer or two integers: `--tile-size 768` or `--tile-size 768 1024`.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=None,
        help="Tile overlap ratio in [0, 1).",
    )
    parser.add_argument(
        "--tile-min-coverage",
        type=float,
        default=None,
        help="Optional minimum edge-tile coverage ratio in [0, 1].",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=None,
        help="Optional tile batch size. Defaults to 1.",
    )
    parser.add_argument(
        "--tile-context-pixels",
        dest="tile_context_pixels",
        nargs="+",
        type=int,
        default=None,
        help="Optional tile context padding as one integer or two integers in pixels.",
    )
    padded_mode_group = parser.add_mutually_exclusive_group()
    padded_mode_group.add_argument(
        "--tile-padded-enabled",
        dest="tile_padded_enabled",
        action="store_true",
        help="Enable padded/context-aware tiled inference.",
    )
    padded_mode_group.add_argument(
        "--tile-padded-disabled",
        dest="tile_padded_enabled",
        action="store_false",
        help="Disable padded/context-aware tiled inference.",
    )
    parser.set_defaults(tile_padded_enabled=None)
    parser.add_argument(
        "--tile-context-size",
        dest="tile_context_size",
        nargs="+",
        type=int,
        default=None,
        help="Padded inference context crop size, e.g. `--tile-context-size 1024`.",
    )
    parser.add_argument(
        "--tile-valid-size",
        dest="tile_valid_size",
        nargs="+",
        type=int,
        default=None,
        help="Padded inference valid center size, e.g. `--tile-valid-size 768`.",
    )
    parser.add_argument(
        "--tile-keep-policy",
        choices=["bbox_center"],
        default=None,
        help="Policy for assigning padded-tile predictions to the valid region.",
    )
    parser.add_argument(
        "--tile-pad-mode",
        choices=["constant", "edge"],
        default=None,
        help="Padding mode for off-image context pixels.",
    )
    parser.add_argument(
        "--tile-pad-value",
        type=int,
        default=None,
        help="Constant pad value when --tile-pad-mode constant is used.",
    )
    border_mode_group = parser.add_mutually_exclusive_group()
    border_mode_group.add_argument(
        "--drop-border-predictions",
        dest="drop_border_predictions",
        action="store_true",
        help="Drop predictions that touch an internal tile border.",
    )
    border_mode_group.add_argument(
        "--keep-border-predictions",
        dest="drop_border_predictions",
        action="store_false",
        help="Keep predictions that touch tile borders.",
    )
    parser.set_defaults(drop_border_predictions=None)
    args = parser.parse_args()

    sweep_runs = _resolve_sweep_runs(
        args.experiment,
        use_config_sweep=args.use_config_sweep,
        score_thr_values=args.score_thr_values,
        max_per_img_values=args.max_per_img_values,
        nms_iou_threshold_values=args.nms_iou_threshold_values,
        max_sweep_runs=args.max_sweep_runs,
    )
    tile_config = resolve_tile_config(
        args.experiment,
        enabled=args.tile_enabled,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        min_tile_coverage=args.tile_min_coverage,
        tile_batch_size=args.tile_batch_size,
        drop_border_predictions=args.drop_border_predictions,
        context_pixels=args.tile_context_pixels,
        padded_enabled=args.tile_padded_enabled,
        padded_context_size=args.tile_context_size,
        padded_valid_size=args.tile_valid_size,
        padded_keep_policy=args.tile_keep_policy,
        padded_pad_mode=args.tile_pad_mode,
        padded_pad_value=args.tile_pad_value,
    )
    tile_merge_postprocess = resolve_tile_merge_postprocess_config(
        args.experiment,
        mask_score_refine_enabled=args.mask_score_refine_enabled,
        mask_score_refine_alpha=args.mask_score_refine_alpha,
        min_mask_area=args.min_mask_area,
    )
    action = "val" if args.split == "val" else "test"
    if len(sweep_runs) == 1:
        effective_output_suffix = None
        if tile_config is not None:
            effective_output_suffix = (
                f"{_build_output_suffix(sweep_runs[0]['score_thr'], sweep_runs[0]['max_per_img'], sweep_runs[0]['nms_iou_threshold'])}"
                f"_{build_tile_output_suffix(tile_config)}"
            )
            effective_output_suffix = append_tile_merge_output_suffix(
                effective_output_suffix,
                tile_merge_postprocess,
            )
        if args.split == "test":
            if tile_config is not None:
                result = run_tiled_inference(
                    split=args.split,
                    experiment_config=args.experiment,
                    checkpoint=args.checkpoint,
                    inference_overrides={
                        **sweep_runs[0],
                        **{
                            key: value
                            for key, value in {
                                "mask_score_refine_enabled": args.mask_score_refine_enabled,
                                "mask_score_refine_alpha": args.mask_score_refine_alpha,
                                "min_mask_area": args.min_mask_area,
                            }.items()
                            if value is not None
                        },
                    },
                    tile_config=tile_config,
                    output_suffix=effective_output_suffix,
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            return run_mmdet_action(
                action,
                args.experiment,
                checkpoint=args.checkpoint,
                inference_overrides=sweep_runs[0],
            )
        sweep_dir = _default_sweep_dir(args.experiment)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        metrics_basename = "val_metrics.json"
        if tile_config is not None:
            metrics_basename = f"val_metrics_{effective_output_suffix}.json"
        metrics_output = args.summary_json or (sweep_dir / metrics_basename)
        if tile_config is not None:
            result = run_tiled_inference(
                split=args.split,
                experiment_config=args.experiment,
                checkpoint=args.checkpoint,
                inference_overrides={
                    **sweep_runs[0],
                    **{
                        key: value
                        for key, value in {
                            "mask_score_refine_enabled": args.mask_score_refine_enabled,
                            "mask_score_refine_alpha": args.mask_score_refine_alpha,
                            "min_mask_area": args.min_mask_area,
                        }.items()
                        if value is not None
                    },
                },
                tile_config=tile_config,
                output_suffix=effective_output_suffix,
                metrics_output=metrics_output,
            )
        else:
            result = run_mmdet_action(
                action,
                args.experiment,
                checkpoint=args.checkpoint,
                inference_overrides=sweep_runs[0],
                metrics_output=metrics_output,
            )
        if isinstance(result, int):
            return result
        print(f"Wrote validation metrics to {metrics_output}")
        return 0

    sweep_dir = _default_sweep_dir(args.experiment)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or (sweep_dir / f"{args.split}_sweep_summary.json")
    summary_runs: list[dict[str, Any]] = []
    for index, sweep_run in enumerate(sweep_runs, start=1):
        output_suffix = _build_output_suffix(
            sweep_run["score_thr"],
            sweep_run["max_per_img"],
            sweep_run["nms_iou_threshold"],
        )
        if tile_config is not None:
            output_suffix = f"{output_suffix}_{build_tile_output_suffix(tile_config)}"
            output_suffix = append_tile_merge_output_suffix(
                output_suffix,
                tile_merge_postprocess,
            )
        print(
            f"[{index}/{len(sweep_runs)}] "
            f"score_thr={sweep_run['score_thr']:.4f}, "
            f"max_per_img={sweep_run['max_per_img']}, "
            f"nms_iou_threshold={sweep_run['nms_iou_threshold']:.4f} "
            f"-> {output_suffix}"
        )
        metrics_output = None
        if args.split == "val":
            metrics_output = sweep_dir / f"val_metrics_{output_suffix}.json"
        if tile_config is not None:
            result = run_tiled_inference(
                split=args.split,
                experiment_config=args.experiment,
                checkpoint=args.checkpoint,
                inference_overrides={
                    **sweep_run,
                    **{
                        key: value
                        for key, value in {
                            "mask_score_refine_enabled": args.mask_score_refine_enabled,
                            "mask_score_refine_alpha": args.mask_score_refine_alpha,
                            "min_mask_area": args.min_mask_area,
                        }.items()
                        if value is not None
                    },
                },
                tile_config=tile_config,
                output_suffix=output_suffix,
                metrics_output=metrics_output,
            )
        else:
            result = run_mmdet_action(
                action,
                args.experiment,
                checkpoint=args.checkpoint,
                inference_overrides=sweep_run,
                output_suffix=output_suffix,
                metrics_output=metrics_output,
            )
        if isinstance(result, int):
            if result != 0:
                return result
            summary_runs.append(
                {
                    **sweep_run,
                    "output_suffix": output_suffix,
                    "prediction_prefix": f"test_release_predictions_{output_suffix}",
                    "tiled": tile_config is not None,
                }
            )
            continue

        summary_runs.append(
            {
                **sweep_run,
                "output_suffix": output_suffix,
                "metrics_json": str(metrics_output.resolve()) if metrics_output else None,
                "metrics": result,
                "tiled": tile_config is not None,
            }
        )
    if summary_runs:
        write_json(
            summary_path,
            {
                "experiment": str(args.experiment),
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "split": args.split,
                "runs": summary_runs,
            },
        )
        print(f"Wrote sweep summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
