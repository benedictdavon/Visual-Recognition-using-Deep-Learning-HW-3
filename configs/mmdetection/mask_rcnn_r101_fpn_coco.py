import os

from src.models.mmdet_mask_rcnn_r101_coco.build import build_mmengine_config_globals


_experiment_config = os.environ.get(
    "HW3_EXPERIMENT_CONFIG",
    "configs/experiments/exp012_mmdet_maskrcnn_r101_coco.yaml",
)

globals().update(build_mmengine_config_globals(_experiment_config))
