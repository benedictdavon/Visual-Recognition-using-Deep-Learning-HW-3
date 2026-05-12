from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.common.config import PROJECT_ROOT, ensure_project_dirs
from src.common.io import read_json, write_json


RECIPE_CONFIGS = {
    "exp015": Path("configs/experiments/exp015_mmdet_maskrcnn_r50_tiled_train_tile_infer.yaml"),
    "exp019": Path("configs/experiments/exp019_r50_all209_tile_only_exp015_exact_scale.yaml"),
}


def _relative_base(config_path: Path, base_config: Path) -> str:
    return os.path.relpath(
        (PROJECT_ROOT / base_config).resolve(),
        start=config_path.resolve().parent,
    )


def _recipe_overrides(
    *,
    recipe: str,
    fold_index: int,
    folds_json: Path,
    tiled_output_root: Path,
) -> dict[str, Any]:
    experiment_name = f"cv_{recipe}_{folds_json.stem}_fold{fold_index}"
    config: dict[str, Any] = {
        "experiment": {
            "id": f"CV-{recipe.upper()}-F{fold_index}",
            "name": experiment_name,
        },
        "runtime": {
            "work_dir": f"outputs/runs/cv/{experiment_name}",
            "disable_validation": False,
            "checkpoint": {
                "interval": 24,
                "save_begin": 0,
                "save_last": True,
                "max_keep_ckpts": 1,
                "save_best": "coco/segm_mAP_50",
                "rule": "greater",
            },
        },
        "dataset": {
            "folds_json": str(folds_json),
            "dev_fold_index": int(fold_index),
            "use_all_train_for_training": False,
            "train_tiled": {
                "enabled": True,
                "tile_size": 768,
                "overlap": 0.0,
                "min_tile_coverage": 0.0,
                "drop_empty_tiles": True,
                "min_visible_fraction": 0.5,
                "min_gt_area": 1.0,
                "force_rebuild": False,
                "output_root": str(tiled_output_root),
            },
        },
    }
    if recipe == "exp019":
        config["experiment"]["notes"] = (
            "Fold-valid exp019-style recipe: all-data training is disabled so the held-out fold "
            "remains clean; the exp019 schedule is inherited."
        )
    return config


def build_cv_plan(
    *,
    folds_json: Path,
    output_config_dir: Path,
    plan_json: Path,
    recipes: list[str],
) -> dict[str, Any]:
    ensure_project_dirs()
    folds_payload = read_json(folds_json)
    folds = folds_payload["folds"]
    output_config_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for recipe in recipes:
        if recipe not in RECIPE_CONFIGS:
            raise ValueError(f"Unknown recipe {recipe!r}; expected one of {sorted(RECIPE_CONFIGS)}.")
        for fold in folds:
            fold_index = int(fold["fold_index"])
            config_path = output_config_dir / f"{recipe}_{folds_json.stem}_fold{fold_index}.yaml"
            tile_root = (
                Path("data/interim/tiled_train")
                / f"{folds_json.stem}_fold{fold_index}_tile768x768_ov0"
            )
            config = {
                "_base_": [_relative_base(config_path, RECIPE_CONFIGS[recipe])],
                **_recipe_overrides(
                    recipe=recipe,
                    fold_index=fold_index,
                    folds_json=folds_json,
                    tiled_output_root=tile_root,
                ),
            }
            write_json(config_path, config)
            work_dir = config["runtime"]["work_dir"]
            runs.append(
                {
                    "recipe": recipe,
                    "fold_index": fold_index,
                    "config": str(config_path),
                    "work_dir": work_dir,
                    "train_command": f"bash scripts/train_exp.sh {config_path}",
                    "tiled_val_command": (
                        "bash scripts/infer_val.sh "
                        f"{config_path} "
                        f"--checkpoint {work_dir}/best_coco_segm_mAP_50_epoch_<N>.pth "
                        "--use-config-sweep"
                    ),
                }
            )

    plan = {
        "folds_json": str(folds_json),
        "strategy": folds_payload.get("strategy"),
        "recipes": recipes,
        "policy": (
            "Use this CV matrix for internal model selection. Public LB should be reserved "
            "for bounded final submissions only."
        ),
        "runs": runs,
    }
    write_json(plan_json, plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create fold-specific configs and commands for exp015/exp019 CV comparison."
    )
    parser.add_argument(
        "--folds-json",
        type=Path,
        default=Path("data/interim/folds/folds_stratified_5.json"),
    )
    parser.add_argument(
        "--output-config-dir",
        type=Path,
        default=Path("configs/experiments/cv"),
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path("outputs/validation/exp015_vs_exp019_stratified5_plan.json"),
    )
    parser.add_argument(
        "--recipe",
        action="append",
        choices=sorted(RECIPE_CONFIGS),
        dest="recipes",
        default=None,
        help="Recipe to include. Repeatable; defaults to exp015 and exp019.",
    )
    args = parser.parse_args()

    plan = build_cv_plan(
        folds_json=args.folds_json,
        output_config_dir=args.output_config_dir,
        plan_json=args.plan_json,
        recipes=args.recipes or ["exp015", "exp019"],
    )
    print(f"Wrote {len(plan['runs'])} CV run configs to {args.output_config_dir}")
    print(f"Wrote CV plan to {args.plan_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
