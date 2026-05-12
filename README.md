# HW3 Instance Segmentation

## Introduction

This repository is for HW3 instance segmentation on colored medical TIFF images. The current first-baseline path is MMDetection with a Mask R-CNN R50-FPN model, using the audited raw data under `data/train`, `data/test_release`, and `data/test_image_name_to_ids.json`.

## Environment Setup

Use the pinned MMDetection setup in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). The short version is:

```bash
bash scripts/setup_mmdet_env.sh
conda run -n hw3-mmdet python -m src.training.launch_mmdet --action check
```

## Usage

Data/config validation only:

```bash
conda run -n hw3-mmdet python -m src.data.validate_mmdet_assets
conda run -n hw3-mmdet python -m src.training.launch_mmdet --action check
```

Training is intentionally not part of the current environment-fix step.

Minimal runtime smoke:

```bash
conda run -n hw3-mmdet python -m src.training.launch_mmdet --action smoke-train
```

One-epoch fold0 baseline:

```bash
conda run -n hw3-mmdet python -m src.training.launch_mmdet --action short-train
```

Test-release inference from a checkpoint:

```bash
conda run -n hw3-mmdet python -m src.inference.predict --checkpoint <checkpoint.pth>
```

Validation inference on the fixed fold0 split:

```bash
conda run -n hw3-mmdet python -m src.inference.predict \
  --split val \
  --experiment configs/experiments/exp001_mmdet_maskrcnn_r50.yaml \
  --checkpoint <checkpoint.pth>
```

Bounded stage-one `max_per_img` sweep on the current experiment config:

```bash
conda run -n hw3-mmdet python -m src.inference.predict \
  --split val \
  --experiment configs/experiments/exp001_mmdet_maskrcnn_r50.yaml \
  --checkpoint <checkpoint.pth> \
  --use-config-sweep
```

Bounded follow-up score-threshold sweep around the chosen `max_per_img`:

```bash
conda run -n hw3-mmdet python -m src.inference.predict \
  --split val \
  --experiment configs/experiments/exp001_mmdet_maskrcnn_r50.yaml \
  --checkpoint <checkpoint.pth> \
  --score-thr 0.03 --score-thr 0.05 --score-thr 0.07 --score-thr 0.10 \
  --max-per-img <best-max-per-img>
```

Build and validate the submission JSON:

```bash
conda run -n hw3-mmdet python -m src.export.build_submission \
  outputs/runs/<experiment-folder>/test_release_predictions.segm.json
conda run -n hw3-mmdet python -m src.export.validate_submission \
  outputs/runs/<experiment-folder>/test-results.json \
  --mapping-json data/test_image_name_to_ids.json
```

The generated JSON filename is always `test-results.json`; the experiment detail lives in the folder name.

## Performance Snapshot

First short fold0 sanity baseline has run for one epoch. It is a pipeline/readiness check, not a tuned model.
