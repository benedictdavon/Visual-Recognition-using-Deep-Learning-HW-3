from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import CONVERTED_DIR, ensure_project_dirs
from src.common.dataset import collect_dataset_stats
from src.common.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute dataset statistics for HW3.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=CONVERTED_DIR / "dataset_stats.json",
        help="Where to write the dataset statistics JSON payload.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    stats = collect_dataset_stats()
    write_json(args.output_json, stats)

    print(f"Wrote dataset stats to {args.output_json}")
    print(
        f"Train={stats['train_sample_count']} Test={stats['test_sample_count']} "
        f"MaskShapeMismatch={stats['mask_shape_mismatch_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
