# Fisher-Hybrid

`Fisher-Hybrid` is a MedGS-based implementation of Fisher-guided density control for medical Gaussian Splatting.

The repository keeps the MedGS training and rendering pipeline, but extends the density-control stage with:

- held-out frame splits for `train`, `fisher-val`, `test`, and `gap`
- a Fisher-style uncertainty proxy computed from image and segmentation losses
- Fisher-guided filtering of densification candidates
- conservative Fisher-guided pruning

This repository is the code package used to reproduce the experiments described in the accompanying report.

## What Is Implemented

The main method-specific files are:

- `train.py`
- `utils/fisher_utils.py`
- `scene/gaussian_model.py`
- `arguments/__init__.py`
- `scene/__init__.py`
- `models/scenes/dataset_readers.py`
- `gaussian_renderer/__init__.py`

The relevant training modes are:

- `--density_mode heuristic`
- `--density_mode fisher_hybrid`

The split-related arguments used in the experiments are:

- `--holdout_stride`
- `--holdout_offset`
- `--second_holdout_offset`
- `--test_split`
- `--gap_start_frac`
- `--gap_end_frac`

## Recommended Workflow

The recommended way to run the repository is with Docker Compose.

The provided `docker-compose.yml` mounts the repository into the container:

```yaml
volumes:
  - .:/workspace
```

This means:

- the image provides the CUDA/PyTorch environment and compiled extensions
- the code is taken from the current working tree through `/workspace`
- host-side edits are visible immediately inside the container
- rebuilding is only required when dependencies, compiled submodules, or the `Dockerfile` change

## Requirements

- NVIDIA GPU
- NVIDIA driver + NVIDIA Container Toolkit
- Docker + Docker Compose

## Clone And Submodules

If you want to rebuild the image locally, clone with submodules:

```bash
git clone --recurse-submodules <repo_url>
cd Fisher-Hybrid
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

The build depends on:

- `submodules/diff-gaussian-rasterization`
- `submodules/simple-knn`

## Docker Usage

### Option 1: Use a prebuilt image

If a prebuilt image has been published to Docker Hub, pull it and run without rebuilding:

```bash
docker pull storiano/fisher-hybrid:cuda12.4
docker compose run --rm --no-build medgs bash
```

If needed, set the image name in `docker-compose.yml` to the published tag.

### Option 2: Build locally

```bash
docker compose build medgs
docker compose run --rm medgs bash
```

## Data Package

The reproduction package is expected to include a ZIP file with the already prepared frame folders.

Example:

```bash
unzip fisher_hybrid_data.zip
```

After extraction, the repository should contain data roots such as:

```text
data/
├── real_014_P3_1_right_img/
│   ├── original/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── mirror/
└── real_014_P3_1_right_seg/
    ├── original/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    └── mirror/
```

In joint training:

- `-s` points to the image root
- `--seg_source_path` points to the segmentation root

Both roots must have:

- the same number of frames
- the same indexing
- matching `original/` and `mirror/` folders

## Split Protocols

### Same-budget

This protocol is used to compare:

- `baseline same-budget`
- `fisher same-budget`

Interpretation:

- `train`: frames used by the main reconstruction loss
- `fisher-val`: held-out frames used only to compute the Fisher proxy and guide density control
- `untouched test`: held-out frames reserved for final evaluation

Reference configuration:

```text
--holdout_stride 8
--holdout_offset 0
--second_holdout_offset 4
--test_split primary
```

This means:

- frames with `idx % 8 == 0` become `fisher-val`
- frames with `idx % 8 == 4` become untouched final test
- all other frames are used by the main reconstruction loss

### Full-budget

This protocol is used for the strongest heuristic baseline.

Interpretation:

- all non-test frames are used for training
- there is no separate `fisher-val` subset

Reference configuration:

```text
--holdout_stride 8
--holdout_offset 4
--second_holdout_offset -1
--test_split primary
```

This means:

- frames with `idx % 8 == 4` are kept as untouched final test
- all remaining frames are used for training

## Reference Training Runs

The three main runs used in the report are listed below.

### 1. Baseline full-budget

This is the strongest heuristic baseline.

```bash
docker compose run --rm --no-build medgs \
  python -u train.py \
  -s data/real_014_P3_1_right_img \
  -m output/expH_014_P3_1_right_baseline_full \
  --pipeline joint \
  --seg_source_path data/real_014_P3_1_right_seg \
  --iterations 30000 \
  --holdout_stride 8 \
  --holdout_offset 4 \
  --second_holdout_offset -1 \
  --test_split primary \
  --density_mode heuristic \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000
```

### 2. Baseline same-budget

This baseline uses the same main reconstruction subset as Fisher, but does not use `fisher-val`.

```bash
docker compose run --rm --no-build medgs \
  python -u train.py \
  -s data/real_014_P3_1_right_img \
  -m output/expA_014_P3_1_right_baseline \
  --pipeline joint \
  --seg_source_path data/real_014_P3_1_right_seg \
  --iterations 30000 \
  --holdout_stride 8 \
  --holdout_offset 0 \
  --second_holdout_offset 4 \
  --test_split primary \
  --density_mode heuristic \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000
```

### 3. Fisher same-budget

This is the proposed Fisher-guided density-control variant.

```bash
docker compose run --rm --no-build medgs \
  python -u train.py \
  -s data/real_014_P3_1_right_img \
  -m output/expA_014_P3_1_right_fisher \
  --pipeline joint \
  --seg_source_path data/real_014_P3_1_right_seg \
  --iterations 30000 \
  --holdout_stride 8 \
  --holdout_offset 0 \
  --second_holdout_offset 4 \
  --test_split primary \
  --density_mode fisher_hybrid \
  --fisher_views_per_update 4 \
  --fisher_ema_decay 0.8 \
  --fisher_weight_xyz 0.5 \
  --fisher_weight_deform 0.5 \
  --fisher_keep_quantile 0.5 \
  --fisher_prune_quantile 0.1 \
  --fisher_prune_opacity 0.05 \
  --fisher_prune_patience 3 \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000
```

## General Training Syntax

The training script still supports the MedGS pipelines:

- `img`
- `seg`
- `joint`

Typical joint training:

```bash
python -u train.py \
  -s <img_dataset_dir> \
  -m <output_dir> \
  --pipeline joint \
  --seg_source_path <seg_dataset_dir>
```

## Rendering

Render a trained model with:

```bash
python render.py --model_path <model_dir> --pipeline both
```

Useful options:

- `--iteration <int>`
- `--pipeline {img,seg,both}`
- `--interp <int>`

## Notes On Reproducibility

- If you use a prebuilt image plus the mounted repository, the image provides the environment and `/workspace` provides the code.
- If you rebuild locally, make sure the submodules are initialized first.
- The `same-budget` protocol should not be interpreted as a full-information baseline: the Fisher variant uses `fisher-val` to guide density control, while the heuristic baseline ignores that subset.

## License And Upstream

This repository is based on MedGS and retains the original upstream license and code structure where applicable. `Fisher-Hybrid` only adds the Fisher-guided density-control extensions on top of that base.
