import os

from src.models.mmdet_mask_rcnn_convnext_t.build import build_mmengine_config_globals


_experiment_config = os.environ.get(
    "HW3_EXPERIMENT_CONFIG",
    "configs/experiments/exp010_mmdet_maskrcnn_convnext_t.yaml",
)

globals().update(build_mmengine_config_globals(_experiment_config))
