from __future__ import annotations

from typing import Any
import warnings

from src.common.io import require_numpy_and_cv2, require_pycocotools_mask


def _ensure_binary_mask(mask):
    np, _ = require_numpy_and_cv2()
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {array.shape}")
    return (array > 0).astype(np.uint8)


def binary_mask_to_uncompressed_rle(mask) -> dict[str, Any]:
    binary_mask = _ensure_binary_mask(mask)
    height, width = binary_mask.shape
    flat = binary_mask.reshape(-1, order="F")
    counts: list[int] = []
    current_value = 0
    run_length = 0
    for value in flat.tolist():
        if value == current_value:
            run_length += 1
            continue
        counts.append(run_length)
        run_length = 1
        current_value = value
    counts.append(run_length)
    return {"size": [int(height), int(width)], "counts": counts}


def uncompressed_rle_to_binary_mask(rle: dict[str, Any]):
    np, _ = require_numpy_and_cv2()
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("RLE size must be a two-item list.")
    if not isinstance(counts, list):
        raise ValueError("Uncompressed RLE counts must be a list.")

    values: list[int] = []
    current_value = 0
    for run_length in counts:
        values.extend([current_value] * int(run_length))
        current_value = 1 - current_value

    mask = np.asarray(values, dtype=np.uint8)
    expected_size = int(size[0]) * int(size[1])
    if mask.size != expected_size:
        raise ValueError(
            f"Decoded mask length {mask.size} does not match expected size {expected_size}"
        )
    return mask.reshape((int(size[0]), int(size[1])), order="F")


def binary_mask_to_compressed_rle(mask) -> dict[str, Any]:
    np, _ = require_numpy_and_cv2()
    mask_utils = require_pycocotools_mask()
    binary_mask = np.asfortranarray(_ensure_binary_mask(mask))
    encoded = mask_utils.encode(binary_mask)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


def compressed_rle_to_binary_mask(rle: dict[str, Any]):
    np, _ = require_numpy_and_cv2()
    mask_utils = require_pycocotools_mask()
    encoded = {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": rle["counts"]}
    if isinstance(encoded["counts"], str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="__array__ implementation doesn't accept a copy keyword",
            category=DeprecationWarning,
        )
        decoded = mask_utils.decode(encoded)
    return np.asarray(decoded, dtype=np.uint8)
