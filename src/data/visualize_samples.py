from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import CLASS_NAMES, SAMPLES_DIR, TRAIN_DIR, ensure_project_dirs
from src.common.dataset import (
    binary_mask_to_bbox,
    collect_dataset_stats,
    iter_binary_instances,
    list_train_samples,
    load_image,
    load_instance_mask,
)
from src.common.io import require_matplotlib, require_numpy_and_cv2, write_json

CLASS_COLORS = {
    "class1": (231, 111, 81),
    "class2": (42, 157, 143),
    "class3": (38, 70, 83),
    "class4": (233, 196, 106),
}


def _to_display_image(image):
    _, cv2 = require_numpy_and_cv2()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported channel count for display: {image.shape}")


def _pick_sample_ids(max_samples: int) -> list[str]:
    stats = collect_dataset_stats()
    ranked = sorted(
        stats["sample_manifest"],
        key=lambda sample: (
            -len(sample["present_classes"]),
            -sample["total_instances"],
            sample["sample_id"],
        ),
    )
    return [sample["sample_id"] for sample in ranked[:max_samples]]


def _load_samples_by_id(sample_ids: list[str]):
    sample_lookup = {sample.sample_id: sample for sample in list_train_samples()}
    return [sample_lookup[sample_id] for sample_id in sample_ids if sample_id in sample_lookup]


def visualize_sample(sample, output_dir: Path) -> Path:
    np, _ = require_numpy_and_cv2()
    plt = require_matplotlib()
    from matplotlib.patches import Rectangle

    image = load_image(sample.image_path)
    display_image = _to_display_image(image)
    height, width = image.shape[:2]

    category_overlay = np.zeros((height, width, 4), dtype=np.float32)
    instance_overlay = np.zeros((height, width, 4), dtype=np.float32)

    instance_counter = 0
    bbox_entries: list[tuple[str, list[float]]] = []
    for class_name in CLASS_NAMES:
        mask_path = sample.class_mask_paths.get(class_name)
        if mask_path is None:
            continue
        mask = load_instance_mask(mask_path)
        union_mask = mask > 0
        if union_mask.any():
            color = np.array(CLASS_COLORS[class_name], dtype=np.float32) / 255.0
            category_overlay[union_mask, :3] = color
            category_overlay[union_mask, 3] = 0.35

        for _, binary_mask in iter_binary_instances(mask):
            instance_counter += 1
            color = np.array(
                [
                    ((instance_counter * 37) % 255) / 255.0,
                    ((instance_counter * 67) % 255) / 255.0,
                    ((instance_counter * 97) % 255) / 255.0,
                ],
                dtype=np.float32,
            )
            instance_overlay[binary_mask, :3] = color
            instance_overlay[binary_mask, 3] = 0.45
            bbox_entries.append((class_name, binary_mask_to_bbox(binary_mask)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].imshow(display_image)
    axes[0].set_title(f"{sample.sample_id} raw image")
    axes[1].imshow(display_image)
    axes[1].imshow(category_overlay)
    axes[1].set_title("Category overlay")
    axes[2].imshow(display_image)
    axes[2].imshow(instance_overlay)
    axes[2].set_title("Per-instance overlay")
    axes[3].imshow(display_image)
    for class_name, bbox in bbox_entries:
        rectangle = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=1.5,
            edgecolor=[value / 255.0 for value in CLASS_COLORS[class_name]],
            facecolor="none",
        )
        axes[3].add_patch(rectangle)
    axes[3].set_title("Bounding boxes")

    for axis in axes:
        axis.set_axis_off()

    figure_path = output_dir / f"{sample.sample_id}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a few train samples with overlays.")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Specific train sample ids to visualize. May be repeated.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=6,
        help="How many samples to visualize when sample ids are not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAMPLES_DIR,
        help="Directory for generated sample figures.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = args.sample_id or _pick_sample_ids(args.max_samples)
    samples = _load_samples_by_id(sample_ids)
    written_paths = [visualize_sample(sample, args.output_dir) for sample in samples]

    manifest_path = args.output_dir / "visualized_samples.json"
    write_json(
        manifest_path,
        {
            "sample_ids": [sample.sample_id for sample in samples],
            "files": [str(path.relative_to(TRAIN_DIR.parent.parent)) for path in written_paths],
        },
    )

    print(f"Wrote {len(written_paths)} sample visualizations to {args.output_dir}")
    print(f"Wrote visualization manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
