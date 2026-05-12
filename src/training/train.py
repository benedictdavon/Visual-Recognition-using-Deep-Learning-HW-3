from __future__ import annotations

import argparse
from pathlib import Path

from src.training.launch_mmdet import run_mmdet_action


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a project experiment.")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/exp001_mmdet_maskrcnn_r50.yaml"),
        help="Experiment-facing config file.",
    )
    parser.add_argument(
        "--smoke-check",
        action="store_true",
        help="Only validate config/data assembly without requiring MMDetection.",
    )
    args = parser.parse_args()
    action = "check" if args.smoke_check else "train"
    return run_mmdet_action(action=action, experiment_config=args.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
