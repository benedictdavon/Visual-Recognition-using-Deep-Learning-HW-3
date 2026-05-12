import os

from src.models.mmdet_rtmdet_mask_head.build import build_mmengine_config_globals


_experiment_config = os.environ.get(
    "HW3_EXPERIMENT_CONFIG",
    "configs/experiments/exp025_mmdet_rtmdet_mask_head_tile_pipeline.yaml",
)

globals().update(build_mmengine_config_globals(_experiment_config))
