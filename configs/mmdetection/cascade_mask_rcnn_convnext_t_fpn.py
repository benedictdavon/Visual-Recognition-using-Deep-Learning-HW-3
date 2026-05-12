import os

from src.models.mmdet_cascade_mask_rcnn_convnext_t.build import build_mmengine_config_globals


_experiment_config = os.environ.get(
    "HW3_EXPERIMENT_CONFIG",
    "configs/experiments/exp032_mmdet_cascade_maskrcnn_convnext_t_tile_pipeline.yaml",
)

globals().update(build_mmengine_config_globals(_experiment_config))
