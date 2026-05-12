import os

from src.models.mmdet_cascade_mask_rcnn_r50_fpn.build import build_mmengine_config_globals


_experiment_config = os.environ.get(
    "HW3_EXPERIMENT_CONFIG",
    "configs/experiments/exp020_mmdet_cascade_maskrcnn_r50_tile_train_tile_infer.yaml",
)

globals().update(build_mmengine_config_globals(_experiment_config))
