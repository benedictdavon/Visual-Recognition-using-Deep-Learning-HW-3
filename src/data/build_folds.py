from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.config import CLASS_NAMES, CONVERTED_DIR, FOLDS_DIR, ensure_project_dirs
from src.common.dataset import collect_dataset_stats
from src.common.io import read_json, write_json

GEOMETRY_TOKEN_PREFIXES = ("min_dim", "area", "aspect")
STRATIFICATION_TOKEN_PREFIXES = (
    "presence",
    "min_dim",
    "area",
    "aspect",
    "count",
    "density",
    "flag",
)


def _token(prefix: str, value: str) -> str:
    return f"{prefix}:{value}"


def _image_area(sample: dict[str, Any]) -> int:
    return int(sample["height"]) * int(sample["width"])


def _total_instances(sample: dict[str, Any]) -> int:
    if "total_instances" in sample:
        return int(sample["total_instances"])
    return int(sum(int(value) for value in sample.get("instance_counts", {}).values()))


def _presence_bits(sample: dict[str, Any]) -> str:
    if sample.get("presence_bits"):
        return str(sample["presence_bits"])
    counts = sample.get("instance_counts", {})
    return "".join("1" if int(counts.get(class_name, 0)) > 0 else "0" for class_name in CLASS_NAMES)


def _min_dim_bin(min_dim: int) -> str:
    if min_dim < 256:
        return "lt256"
    if min_dim < 512:
        return "256_511"
    if min_dim < 768:
        return "512_767"
    return "gte768"


def _area_bin(area_pixels: int) -> str:
    megapixels = area_pixels / 1_000_000.0
    if megapixels < 0.10:
        return "lt0p10mp"
    if megapixels < 0.30:
        return "0p10_0p30mp"
    if megapixels < 0.75:
        return "0p30_0p75mp"
    if megapixels < 1.50:
        return "0p75_1p50mp"
    return "gte1p50mp"


def _aspect_bin(aspect_ratio: float) -> str:
    if aspect_ratio < 1.25:
        return "lt1p25"
    if aspect_ratio < 1.75:
        return "1p25_1p75"
    if aspect_ratio < 2.50:
        return "1p75_2p50"
    return "gte2p50"


def _count_bin(total_instances: int) -> str:
    if total_instances <= 10:
        return "le10"
    if total_instances <= 50:
        return "11_50"
    if total_instances <= 150:
        return "51_150"
    if total_instances <= 300:
        return "151_300"
    return "gt300"


def _density_bin(instances_per_megapixel: float) -> str:
    if instances_per_megapixel < 50:
        return "lt50"
    if instances_per_megapixel < 150:
        return "50_150"
    if instances_per_megapixel < 300:
        return "150_300"
    if instances_per_megapixel < 600:
        return "300_600"
    return "gte600"


def _feature_tokens_for_image(
    *,
    height: int,
    width: int,
    total_instances: int | None = None,
    presence_bits: str | None = None,
) -> list[str]:
    min_dim = min(height, width)
    max_dim = max(height, width)
    area = height * width
    aspect = max_dim / max(1.0, float(min_dim))
    tokens = [
        _token("min_dim", _min_dim_bin(min_dim)),
        _token("area", _area_bin(area)),
        _token("aspect", _aspect_bin(aspect)),
    ]
    if presence_bits is not None:
        tokens.append(_token("presence", presence_bits))
    if total_instances is not None:
        megapixels = max(area / 1_000_000.0, 1e-9)
        tokens.extend(
            [
                _token("count", _count_bin(total_instances)),
                _token("density", _density_bin(total_instances / megapixels)),
            ]
        )
    if min_dim < 256:
        tokens.append(_token("flag", "tiny_min_dim"))
    if aspect >= 2.0:
        tokens.append(_token("flag", "extreme_aspect"))
    return tokens


def _enrich_sample(sample: dict[str, Any]) -> dict[str, Any]:
    height = int(sample["height"])
    width = int(sample["width"])
    total_instances = _total_instances(sample)
    area = _image_area(sample)
    min_dim = min(height, width)
    max_dim = max(height, width)
    aspect = max_dim / max(1.0, float(min_dim))
    megapixels = max(area / 1_000_000.0, 1e-9)
    enriched = dict(sample)
    enriched["_total_instances"] = total_instances
    enriched["_area_pixels"] = area
    enriched["_min_dim"] = min_dim
    enriched["_aspect_ratio"] = aspect
    enriched["_instances_per_megapixel"] = total_instances / megapixels
    enriched["_presence_bits"] = _presence_bits(sample)
    enriched["_feature_tokens"] = _feature_tokens_for_image(
        height=height,
        width=width,
        total_instances=total_instances,
        presence_bits=enriched["_presence_bits"],
    )
    return enriched


def _summarize_numeric(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "mean": 0, "max": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _summarize_enriched_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts: Counter[str] = Counter()
    class_presence = Counter({class_name: 0 for class_name in CLASS_NAMES})
    class_instances = Counter({class_name: 0 for class_name in CLASS_NAMES})
    for sample in samples:
        token_counts.update(sample["_feature_tokens"])
        counts = sample.get("instance_counts", {})
        for class_name in CLASS_NAMES:
            count = int(counts.get(class_name, 0))
            if count > 0:
                class_presence[class_name] += 1
            class_instances[class_name] += count
    return {
        "sample_count": len(samples),
        "class_presence": dict(class_presence),
        "class_instances": dict(class_instances),
        "feature_counts": dict(sorted(token_counts.items())),
        "image_area_pixels": _summarize_numeric([sample["_area_pixels"] for sample in samples]),
        "min_dim": _summarize_numeric([sample["_min_dim"] for sample in samples]),
        "aspect_ratio": _summarize_numeric([sample["_aspect_ratio"] for sample in samples]),
        "total_instances": _summarize_numeric([sample["_total_instances"] for sample in samples]),
        "instances_per_megapixel": _summarize_numeric(
            [sample["_instances_per_megapixel"] for sample in samples]
        ),
    }


def _feature_counter(samples: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for sample in samples:
        counter.update(sample["_feature_tokens"])
    return counter


def _token_weight(token: str) -> float:
    prefix = token.split(":", 1)[0]
    if prefix == "flag":
        return 6.0
    if prefix in {"min_dim", "aspect", "area"}:
        return 4.0
    if prefix in {"count", "density"}:
        return 2.0
    if prefix == "presence":
        return 1.5
    return 1.0


def build_stratified_folds(sample_manifest: list[dict[str, Any]], num_folds: int) -> dict[str, Any]:
    if num_folds < 3 or num_folds > 5:
        raise ValueError("Stratified validation is intended for 3-5 folds.")

    samples = [_enrich_sample(sample) for sample in sample_manifest]
    global_features = _feature_counter(samples)
    global_summary = _summarize_enriched_samples(samples)
    target_sample_count = len(samples) / float(num_folds)
    target_features = {
        token: count / float(num_folds)
        for token, count in global_features.items()
    }
    global_class_instances = global_summary["class_instances"]
    target_class_instances = {
        class_name: max(1.0, float(global_class_instances[class_name]) / float(num_folds))
        for class_name in CLASS_NAMES
    }

    fold_states = [
        {
            "fold_index": fold_index,
            "samples": [],
            "feature_counts": Counter(),
            "class_instances": Counter({class_name: 0 for class_name in CLASS_NAMES}),
        }
        for fold_index in range(num_folds)
    ]

    ordered_samples = sorted(
        samples,
        key=lambda sample: (
            _token("flag", "tiny_min_dim") not in sample["_feature_tokens"],
            _token("flag", "extreme_aspect") not in sample["_feature_tokens"],
            -sum(1 for class_name in ("class3", "class4") if sample["instance_counts"].get(class_name, 0) > 0),
            -sample["_total_instances"],
            -sample["_instances_per_megapixel"],
            sample["sample_id"],
        ),
    )

    def assignment_score(fold_state: dict[str, Any], sample: dict[str, Any]) -> tuple[float, int, int]:
        next_count = len(fold_state["samples"]) + 1
        # Keep fold sizes close first; feature balance is only useful when every
        # fold is a real holdout rather than an overfit geometry bucket.
        score = (next_count / max(1.0, target_sample_count)) ** 2 * 100.0
        if next_count > target_sample_count + 1:
            score += ((next_count - target_sample_count) ** 2) * 10.0
        for token in sample["_feature_tokens"]:
            current = float(fold_state["feature_counts"][token])
            target = max(1.0, target_features.get(token, 1.0))
            excess = max(0.0, current + 1.0 - target)
            score += (excess / target) ** 2 * _token_weight(token)
        for class_name in CLASS_NAMES:
            added = float(sample.get("instance_counts", {}).get(class_name, 0))
            if added <= 0:
                continue
            target = target_class_instances[class_name]
            current = float(fold_state["class_instances"][class_name])
            excess = max(0.0, current + added - target)
            score += (excess / target) ** 2
        return (score, next_count, int(fold_state["fold_index"]))

    for sample in ordered_samples:
        best_fold = min(fold_states, key=lambda fold_state: assignment_score(fold_state, sample))
        best_fold["samples"].append(sample)
        best_fold["feature_counts"].update(sample["_feature_tokens"])
        for class_name in CLASS_NAMES:
            best_fold["class_instances"][class_name] += int(
                sample.get("instance_counts", {}).get(class_name, 0)
            )

    all_sample_ids = [sample["sample_id"] for sample in samples]
    folds = []
    for fold_state in fold_states:
        val_samples = sorted(fold_state["samples"], key=lambda sample: sample["sample_id"])
        val_ids = [sample["sample_id"] for sample in val_samples]
        val_id_set = set(val_ids)
        train_ids = [sample_id for sample_id in all_sample_ids if sample_id not in val_id_set]
        folds.append(
            {
                "fold_index": fold_state["fold_index"],
                "train_sample_ids": train_ids,
                "val_sample_ids": val_ids,
                "summary": _summarize_enriched_samples(val_samples),
            }
        )

    return {
        "num_folds": num_folds,
        "strategy": "stratified_by_class_presence_size_tiny_aspect_density_v1",
        "warning": (
            "Use these folds for model selection diagnostics. Public leaderboard should be "
            "reserved for bounded final submissions, not broad tuning."
        ),
        "feature_definitions": {
            "tiny_min_dim": "min(height, width) < 256",
            "extreme_aspect": "max(height, width) / min(height, width) >= 2.0",
            "density": "instances per megapixel, binned from train annotations",
        },
        "global_summary": global_summary,
        "folds": folds,
    }


def _test_image_records(test_image_info_payload: Any) -> list[dict[str, Any]]:
    if isinstance(test_image_info_payload, dict):
        return list(test_image_info_payload.get("images", []))
    if isinstance(test_image_info_payload, list):
        return list(test_image_info_payload)
    raise ValueError("Test image info must be either a COCO-like dict or a mapping list.")


def _geometry_tokens_for_record(record: dict[str, Any]) -> list[str]:
    return _feature_tokens_for_image(
        height=int(record["height"]),
        width=int(record["width"]),
        total_instances=None,
        presence_bits=None,
    )


def build_test_like_split(
    sample_manifest: list[dict[str, Any]],
    test_image_info_payload: Any,
    *,
    val_count: int = 42,
) -> dict[str, Any]:
    samples = [_enrich_sample(sample) for sample in sample_manifest]
    if val_count <= 0 or val_count >= len(samples):
        raise ValueError(f"val_count must be within [1, {len(samples) - 1}], got {val_count}.")

    test_records = _test_image_records(test_image_info_payload)
    test_geometry_counts: Counter[str] = Counter()
    for record in test_records:
        test_geometry_counts.update(_geometry_tokens_for_record(record))

    train_summary = _summarize_enriched_samples(samples)
    train_feature_counts = _feature_counter(samples)
    geometry_targets = {
        token: (count / max(1, len(test_records))) * float(val_count)
        for token, count in test_geometry_counts.items()
        if token.split(":", 1)[0] in GEOMETRY_TOKEN_PREFIXES or token.startswith("flag:")
    }
    train_targets = {
        token: (count / float(len(samples))) * float(val_count)
        for token, count in train_feature_counts.items()
        if token.split(":", 1)[0] not in GEOMETRY_TOKEN_PREFIXES
    }
    target_features = {**train_targets, **geometry_targets}

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_features: Counter[str] = Counter()

    def selection_score(sample: dict[str, Any]) -> tuple[float, str]:
        score = 0.0
        sample_tokens = set(sample["_feature_tokens"])
        candidate_tokens = set(target_features) | sample_tokens
        for token in candidate_tokens:
            target = max(0.001, target_features.get(token, 0.0))
            current = float(selected_features[token])
            added = 1.0 if token in sample_tokens else 0.0
            weight = _token_weight(token)
            score += ((current + added - target) / max(1.0, target)) ** 2 * weight
        return (score, sample["sample_id"])

    while len(selected) < val_count:
        candidates = [sample for sample in samples if sample["sample_id"] not in selected_ids]
        best_sample = min(candidates, key=selection_score)
        selected.append(best_sample)
        selected_ids.add(best_sample["sample_id"])
        selected_features.update(best_sample["_feature_tokens"])

    selected = sorted(selected, key=lambda sample: sample["sample_id"])
    val_ids = [sample["sample_id"] for sample in selected]
    val_id_set = set(val_ids)
    train_ids = [sample["sample_id"] for sample in samples if sample["sample_id"] not in val_id_set]

    return {
        "num_folds": 1,
        "strategy": "test_like_geometry_holdout_v1",
        "warning": (
            "This is an internal diagnostic split biased toward test-like image geometry. "
            "It is not a public leaderboard substitute."
        ),
        "test_geometry_summary": {
            "image_count": len(test_records),
            "feature_counts": dict(sorted(test_geometry_counts.items())),
        },
        "global_summary": train_summary,
        "folds": [
            {
                "fold_index": 0,
                "train_sample_ids": train_ids,
                "val_sample_ids": val_ids,
                "summary": _summarize_enriched_samples(selected),
            }
        ],
    }


def build_folds(sample_manifest: list[dict[str, Any]], num_folds: int) -> dict[str, Any]:
    fold_states = [
        {
            "fold_index": fold_index,
            "sample_ids": [],
            "sample_count": 0,
            "class_presence": {class_name: 0 for class_name in CLASS_NAMES},
            "class_instances": {class_name: 0 for class_name in CLASS_NAMES},
        }
        for fold_index in range(num_folds)
    ]

    ordered_samples = sorted(
        sample_manifest,
        key=lambda sample: (
            -sum(1 for class_name in CLASS_NAMES if sample["instance_counts"].get(class_name, 0) > 0),
            -sum(sample["instance_counts"].values()),
            sample["sample_id"],
        ),
    )

    for sample_index, sample in enumerate(ordered_samples):
        best_fold = fold_states[sample_index % num_folds]
        best_fold["sample_ids"].append(sample["sample_id"])
        best_fold["sample_count"] += 1
        for class_name in CLASS_NAMES:
            if sample["instance_counts"].get(class_name, 0) > 0:
                best_fold["class_presence"][class_name] += 1
            best_fold["class_instances"][class_name] += sample["instance_counts"].get(class_name, 0)

    all_sample_ids = [sample["sample_id"] for sample in sample_manifest]
    folds = []
    for fold_state in fold_states:
        val_ids = sorted(fold_state["sample_ids"])
        val_id_set = set(val_ids)
        train_ids = [sample_id for sample_id in all_sample_ids if sample_id not in val_id_set]
        folds.append(
            {
                "fold_index": fold_state["fold_index"],
                "train_sample_ids": train_ids,
                "val_sample_ids": val_ids,
                "summary": {
                    "sample_count": fold_state["sample_count"],
                    "class_presence": fold_state["class_presence"],
                    "class_instances": fold_state["class_instances"],
                },
            }
        )

    return {
        "num_folds": num_folds,
        "strategy": "round_robin_after_sort_by_presence_and_instance_count",
        "warning": "Fold generation is a data-side scaffold only; validation policy is still open in docs.",
        "folds": folds,
    }


def load_fold_payload(folds_json: Path) -> dict[str, Any]:
    payload = read_json(folds_json)
    if not isinstance(payload, dict) or "folds" not in payload:
        raise ValueError(f"Invalid folds payload: {folds_json}")
    return payload


def _subset_coco_by_sample_ids(
    coco_payload: dict[str, Any],
    sample_ids: set[str],
) -> dict[str, Any]:
    images = [
        image for image in coco_payload["images"] if image["sample_id"] in sample_ids
    ]
    image_id_set = {image["id"] for image in images}
    annotations = [
        annotation
        for annotation in coco_payload["annotations"]
        if annotation["image_id"] in image_id_set
    ]
    return {
        "info": coco_payload["info"],
        "licenses": coco_payload["licenses"],
        "images": images,
        "annotations": annotations,
        "categories": coco_payload["categories"],
    }


def _summarize_subset(coco_payload: dict[str, Any]) -> dict[str, Any]:
    category_by_id = {
        category["id"]: category["name"] for category in coco_payload["categories"]
    }
    class_presence = Counter({class_name: 0 for class_name in CLASS_NAMES})
    instance_counts = Counter({class_name: 0 for class_name in CLASS_NAMES})

    image_ids_by_class: dict[str, set[int]] = {class_name: set() for class_name in CLASS_NAMES}
    for annotation in coco_payload["annotations"]:
        class_name = category_by_id[int(annotation["category_id"])]
        image_ids_by_class[class_name].add(int(annotation["image_id"]))
        instance_counts[class_name] += 1

    for class_name in CLASS_NAMES:
        class_presence[class_name] = len(image_ids_by_class[class_name])

    return {
        "image_count": len(coco_payload["images"]),
        "annotation_count": len(coco_payload["annotations"]),
        "class_presence": dict(class_presence),
        "class_instances": dict(instance_counts),
    }


def materialize_dev_split_assets(
    full_train_json: Path = CONVERTED_DIR / "instances_train.json",
    folds_json: Path = FOLDS_DIR / "folds_5.json",
    fold_index: int = 0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_project_dirs()
    folds_payload = load_fold_payload(folds_json)
    matching_folds = [
        fold for fold in folds_payload["folds"] if int(fold["fold_index"]) == int(fold_index)
    ]
    if not matching_folds:
        raise ValueError(f"Fold index {fold_index} is not present in {folds_json}")

    fold_record = matching_folds[0]
    train_sample_ids = set(fold_record["train_sample_ids"])
    val_sample_ids = set(fold_record["val_sample_ids"])
    if train_sample_ids & val_sample_ids:
        raise ValueError(f"Fold {fold_index} has overlapping train/val sample ids.")

    coco_payload = read_json(full_train_json)
    if output_dir is not None:
        output_root = output_dir
    elif folds_json.resolve() == (FOLDS_DIR / "folds_5.json").resolve():
        output_root = FOLDS_DIR / f"dev_split_fold{fold_index}"
    else:
        output_root = FOLDS_DIR / f"{folds_json.stem}_fold{fold_index}"
    output_root.mkdir(parents=True, exist_ok=True)

    train_subset = _subset_coco_by_sample_ids(coco_payload, train_sample_ids)
    val_subset = _subset_coco_by_sample_ids(coco_payload, val_sample_ids)

    train_json = output_root / f"instances_fold{fold_index}_train.json"
    val_json = output_root / f"instances_fold{fold_index}_val.json"
    train_ids_json = output_root / f"fold{fold_index}_train_sample_ids.json"
    val_ids_json = output_root / f"fold{fold_index}_val_sample_ids.json"
    split_summary_json = output_root / f"fold{fold_index}_summary.json"

    split_summary = {
        "fold_index": fold_index,
        "strategy": folds_payload.get("strategy"),
        "train": _summarize_subset(train_subset),
        "val": _summarize_subset(val_subset),
    }

    write_json(train_json, train_subset)
    write_json(val_json, val_subset)
    write_json(train_ids_json, sorted(train_sample_ids))
    write_json(val_ids_json, sorted(val_sample_ids))
    write_json(split_summary_json, split_summary)

    return {
        "fold_index": fold_index,
        "train_ann_file": str(train_json),
        "val_ann_file": str(val_json),
        "train_sample_ids_file": str(train_ids_json),
        "val_sample_ids_file": str(val_ids_json),
        "summary_file": str(split_summary_json),
        "train_image_count": split_summary["train"]["image_count"],
        "val_image_count": split_summary["val"]["image_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic scaffold folds for HW3.")
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=CONVERTED_DIR / "train_sample_manifest.json",
        help="Path to the train sample manifest generated by convert_to_coco.py.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="How many folds to generate.",
    )
    parser.add_argument(
        "--strategy",
        choices=["legacy", "stratified"],
        default="legacy",
        help="Fold assignment strategy. `legacy` preserves the original scaffold.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=FOLDS_DIR / "folds_5.json",
        help="Where to write the fold assignments.",
    )
    parser.add_argument(
        "--test-image-info-json",
        type=Path,
        default=CONVERTED_DIR / "image_info_test_release.json",
        help="COCO-like test image info used for test-like split geometry targets.",
    )
    parser.add_argument(
        "--materialize-dev-split",
        action="store_true",
        help="Also emit split-specific COCO JSON assets for one fixed dev fold.",
    )
    parser.add_argument(
        "--materialize-all-folds",
        action="store_true",
        help="Emit split-specific COCO JSON assets for every fold in the output fold file.",
    )
    parser.add_argument(
        "--dev-fold-index",
        type=int,
        default=0,
        help="Which fold to materialize when --materialize-dev-split is set.",
    )
    parser.add_argument(
        "--build-test-like-split",
        action="store_true",
        help="Also build a single held-out split biased toward test-release image geometry.",
    )
    parser.add_argument(
        "--test-like-val-count",
        type=int,
        default=42,
        help="Validation image count for the test-like split.",
    )
    parser.add_argument(
        "--test-like-output-json",
        type=Path,
        default=FOLDS_DIR / "test_like_split_42.json",
        help="Where to write the test-like split assignment.",
    )
    parser.add_argument(
        "--materialize-test-like-split",
        action="store_true",
        help="Emit split-specific COCO JSON assets for the test-like split.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    if args.manifest_json.exists():
        sample_manifest = read_json(args.manifest_json)
    else:
        stats = collect_dataset_stats()
        sample_manifest = stats["sample_manifest"]

    if args.strategy == "stratified":
        folds = build_stratified_folds(sample_manifest, args.num_folds)
    else:
        folds = build_folds(sample_manifest, args.num_folds)
    write_json(args.output_json, folds)
    print(f"Wrote {args.num_folds} fold assignments to {args.output_json}")
    if args.materialize_dev_split:
        split_assets = materialize_dev_split_assets(
            full_train_json=CONVERTED_DIR / "instances_train.json",
            folds_json=args.output_json,
            fold_index=args.dev_fold_index,
        )
        print(
            f"Materialized dev split fold {args.dev_fold_index}: "
            f"train={split_assets['train_image_count']} val={split_assets['val_image_count']}"
        )
    if args.materialize_all_folds:
        for fold in folds["folds"]:
            split_assets = materialize_dev_split_assets(
                full_train_json=CONVERTED_DIR / "instances_train.json",
                folds_json=args.output_json,
                fold_index=int(fold["fold_index"]),
            )
            print(
                f"Materialized fold {fold['fold_index']}: "
                f"train={split_assets['train_image_count']} val={split_assets['val_image_count']}"
            )
    if args.build_test_like_split:
        test_like_payload = build_test_like_split(
            sample_manifest,
            read_json(args.test_image_info_json),
            val_count=int(args.test_like_val_count),
        )
        write_json(args.test_like_output_json, test_like_payload)
        print(
            f"Wrote test-like split with {args.test_like_val_count} validation images "
            f"to {args.test_like_output_json}"
        )
        if args.materialize_test_like_split:
            split_assets = materialize_dev_split_assets(
                full_train_json=CONVERTED_DIR / "instances_train.json",
                folds_json=args.test_like_output_json,
                fold_index=0,
            )
            print(
                "Materialized test-like split: "
                f"train={split_assets['train_image_count']} val={split_assets['val_image_count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
