from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import LOGS_DIR, ensure_project_dirs
from src.common.dataset import collect_dataset_stats, format_audit_summary
from src.common.io import write_json, write_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the live HW3 dataset layout.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=LOGS_DIR / "dataset_audit_summary.json",
        help="Path for the machine-readable audit summary.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=LOGS_DIR / "dataset_audit_summary.md",
        help="Path for the short markdown audit summary.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    stats = collect_dataset_stats()
    summary = format_audit_summary(stats)

    write_json(args.summary_json, stats)
    write_text(args.summary_md, summary)

    print(summary.rstrip())
    print(f"\nWrote audit JSON to {args.summary_json}")
    print(f"Wrote audit markdown to {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
