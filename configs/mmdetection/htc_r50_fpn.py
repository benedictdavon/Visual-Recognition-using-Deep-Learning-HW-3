import os

from src.models.mmdet_htc_r50_fpn.build import build_mmengine_config_globals


_experiment_config = os.environ.get(
    "HW3_EXPERIMENT_CONFIG",
    "configs/experiments/exp022_mmdet_htc_r50_tile_train_tile_infer.yaml",
)

globals().update(build_mmengine_config_globals(_experiment_config))
