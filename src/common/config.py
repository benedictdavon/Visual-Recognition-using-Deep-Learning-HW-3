from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_RELEASE_DIR = DATA_DIR / "test_release"
TEST_IMAGE_ID_MAP_PATH = DATA_DIR / "test_image_name_to_ids.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"
CONFIGS_BASE_DIR = CONFIGS_DIR / "base"
CONFIGS_EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"
MMDET_CONFIGS_DIR = CONFIGS_DIR / "mmdetection"

INTERIM_DIR = DATA_DIR / "interim"
CONVERTED_DIR = INTERIM_DIR / "converted"
FOLDS_DIR = INTERIM_DIR / "folds"
PSEUDO_LABELS_DIR = INTERIM_DIR / "pseudo_labels"
SAMPLES_DIR = DATA_DIR / "samples"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RUNS_DIR = OUTPUTS_DIR / "runs"
CONFIG_ARTIFACTS_DIR = OUTPUTS_DIR / "generated_configs"

CLASS_NAMES = tuple(f"class{index}" for index in range(1, 5))
CATEGORY_ID_BY_NAME = {name: index for index, name in enumerate(CLASS_NAMES, start=1)}
CATEGORIES = [
    {"id": category_id, "name": class_name, "supercategory": "cell"}
    for class_name, category_id in CATEGORY_ID_BY_NAME.items()
]
CLASS_ORDER_BITS = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}

DEFAULT_RUNTIME_HINT = "Run this command with `conda run -n hw3-mmdet ...`."


def ensure_project_dirs() -> None:
    """Create the minimal scaffold this phase depends on."""
    for path in (
        CONVERTED_DIR,
        FOLDS_DIR,
        PSEUDO_LABELS_DIR,
        SAMPLES_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        RUNS_DIR,
        CONFIG_ARTIFACTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
