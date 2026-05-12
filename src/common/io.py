from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.common.config import DEFAULT_RUNTIME_HINT


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def require_numpy_and_cv2():
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "This utility needs `numpy` and `opencv-python`. "
            f"{DEFAULT_RUNTIME_HINT}"
        ) from error
    logging_module = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging_module is not None:
        try:
            logging_module.setLogLevel(logging_module.LOG_LEVEL_ERROR)
        except Exception:
            pass
    return np, cv2


def require_matplotlib():
    cache_root = Path("/tmp/hw3_matplotlib_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "This utility needs `matplotlib`. "
            f"{DEFAULT_RUNTIME_HINT}"
        ) from error
    return plt


def require_pycocotools_mask():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "This utility needs `pycocotools`. "
            f"{DEFAULT_RUNTIME_HINT}"
        ) from error
    return mask_utils
