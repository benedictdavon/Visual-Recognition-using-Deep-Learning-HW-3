import os

from src.models.mmdet_mask_rcnn_r50.build import build_mmengine_config_globals


_experiment_config = os.environ.get(
    "HW3_EXPERIMENT_CONFIG",
    "configs/experiments/exp001_mmdet_maskrcnn_r50.yaml",
)

globals().update(build_mmengine_config_globals(_experiment_config))
