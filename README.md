# segmentation-tools

A CLI pipeline for aligning and segmenting multiplexed microscopy images. Registers a moving image (e.g., IF/immunofluorescence) to a fixed reference (e.g., Xenium spatial transcriptomics) using VALIS for global affine alignment followed by MIRAGE for local non-linear correction, then segments cells with CellPose.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Config file (recommended)](#config-file-recommended)
  - [CLI flags](#cli-flags)
  - [Submitting to SLURM](#submitting-to-slurm)
  - [Resuming a run](#resuming-a-run)
  - [Viewing results](#viewing-results)
- [Pipeline Steps](#pipeline-steps)
- [Output Structure](#output-structure)
- [Requirements](#requirements)

---

## Installation

```bash
# Clone and install in editable mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With VALIS alignment support (required for this branch)
pip install -e ".[valis]"
```

> **Note:** MIRAGE (in `src/MIRAGE/`) is a separate package using Poetry. It requires TensorFlow and a GPU.
> ```bash
> cd src/MIRAGE && poetry install
> ```

---

## Quick Start

**1. Generate a config file:**
```bash
segmentation-tools init-config > my_sample.yaml
```

**2. Edit `my_sample.yaml`** with your file paths and channel indices:
```yaml
job_title: "sample_001"
fixed_file: "/path/to/xenium/morphology_focus_0000.ome.tif"
moving_file: "/path/to/if/Region_1.nd2"
output_root: "/path/to/output"
fixed_dapi_channel: 0
moving_dapi_channel: 1
```

**3. Submit to SLURM:**
```bash
segmentation-tools submit --config my_sample.yaml
```

**4. View QC report when done:**
```bash
segmentation-tools view -o /path/to/output -j sample_001
```

---

## Usage

### Config file (recommended)

Generate a sample config and edit it:
```bash
segmentation-tools init-config > my_sample.yaml
segmentation-tools submit --config my_sample.yaml
```

The config covers all inputs, channel indices, MIRAGE hyperparameters, and SLURM settings. See [config_example.yaml](src/segmentation_tools/config_example.yaml) for the full reference.

### CLI flags

All config values can be specified directly as CLI flags (these override config file values):

```bash
segmentation-tools run \
  -f /path/to/fixed.ome.tif \
  -m /path/to/moving.nd2 \
  -o /path/to/output \
  -j my_sample \
  --fixed-dapi-channel 0 \
  --moving-dapi-channel 1
```

### Submitting to SLURM

```bash
# From config file
segmentation-tools submit --config my_sample.yaml

# With SLURM overrides
segmentation-tools submit --config my_sample.yaml \
  --partition gpu --mem 200G --cpus 8 --time 12:00:00

# Dry run — print the sbatch command without submitting
segmentation-tools submit --config my_sample.yaml --dry-run
```

Default SLURM settings (overridable via config or flags):
| Setting | Default |
|---------|---------|
| `conda_env` | `contamination` |
| `partition` | `peerd` |
| `time` | `23:00:00` |
| `mem` | `500G` |
| `gpus` | `1` |
| `cpus` | `2` |

### Running locally

```bash
conda activate contamination
segmentation-tools run --config my_sample.yaml
```

### Resuming a run

Each step has a number. Pass `--start-step N` to skip all steps before N:

```bash
# Re-run from VALIS alignment onward (skips TIFF conversion)
segmentation-tools submit --config my_sample.yaml --start-step 2

# Re-run from MIRAGE onward (skips alignment entirely)
segmentation-tools submit --config my_sample.yaml --start-step 6

# Can also override in the config file
start_step: 4
```

| Step | Number |
|------|--------|
| Setup directories + convert to OME-TIFF | `1` |
| VALIS rigid/affine alignment | `2` |
| Preprocess high-res DAPI | `3` |
| Apply linear transform to moving DAPI | `4` |
| Recommend MIRAGE hyperparameters | `5` |
| MIRAGE non-linear registration | `6` |
| Evaluate MIRAGE alignment quality | `7` |
| Warp all channels + build pyramid OME-TIFF | `8` |
| CellPose segmentation + masks to parquet | `9` |

### Viewing results

After a run completes (or mid-run), generate a QC HTML report:

```bash
segmentation-tools view -o /path/to/output -j my_sample
# Opens results/qc_report.html in your browser
# Use --no-browser to just write the file
```

The report includes:
- Pipeline status and last completed step
- Registration quality metrics (rotation angle, scale, translation, centroid distances before/after MIRAGE)
- Cell counts per segmentation mode
- Alignment overlay images (VALIS before/after, pre-MIRAGE overlay, displacement quiver plots)

---

## Pipeline Steps

### Step 1: Convert to TIFF
Converts the fixed and moving images to OME-TIFF using `bioformats2raw` + `raw2ometiff`. Accepts most microscopy formats (.nd2, .czi, .mrxs, .tiff, etc.).

**Outputs:** `.checkpoints/fixed.tiff`, `.checkpoints/moving.tiff`

### Step 2: VALIS Alignment
Runs VALIS rigid/affine registration on the DAPI channels of both images. Handles cross-modality registration (fluorescence vs fluorescence), including automatic detection of reflections and rotations. Non-rigid registration is disabled — MIRAGE handles that.

**Outputs:** `.checkpoints/linear_transform.npy` (3×3 inverse map matrix)

### Step 3: Preprocess High-Res DAPI
Extracts the DAPI channel at full resolution from both images, applies quantile normalization (1st–99th percentile clip) + CLAHE + multi-Otsu threshold. These preprocessed images are used as MIRAGE inputs.

**Outputs:** `.checkpoints/high_res_fixed_dapi_filtered_level_0.npy`, `.checkpoints/high_res_moving_dapi_filtered_level_0.npy`

### Step 4: Warp Moving DAPI
Applies the VALIS affine transform to the preprocessed high-res moving DAPI image. This linearly-warped image is the starting point for MIRAGE.

**Output:** `.checkpoints/moving_dapi_linear_warped.npy`

### Step 5: Recommend MIRAGE Parameters
Samples 5 tissue crops (1000×1000 px), extracts nuclei centroids, matches them between fixed and moving images via KDTree nearest-neighbor, and recommends MIRAGE hyperparameters (offset, pad, smoothness_radius, pos_encoding_L, dissim_sigma) based on the displacement statistics and nucleus size.

**Outputs:** `.checkpoints/recommended_mirage_params.npy`, `results/centroid_quiver_crop_*.png`, `results/mirage_param_recommendation.png`

### Step 6: MIRAGE Non-Linear Registration
Trains a coordinate-based neural network (TensorFlow) to predict a dense per-pixel displacement field that corrects residual non-linear distortions after the affine alignment. Automatically loads the parameters recommended by step 5b.

**Output:** `.checkpoints/mirage_transform.npy`

### Step 6b: Evaluate MIRAGE Alignment
Applies the MIRAGE transform to the same tissue crops used for parameter recommendation and computes centroid matching distances before and after correction. Generates before/after overlay plots.

**Outputs:** `results/centroid_eval_crop_*.png`, `results/centroid_eval_summary.png`

### Step 7: Warp All Channels
Combines the VALIS linear transform and MIRAGE displacement field, then applies the composite warp to all channels of the moving image. Builds a multi-resolution OME-TIFF pyramid using pyvips.

**Output:** `results/moving_complete_transform.ome.tiff`

### Step 8: CellPose Segmentation
Runs CellPose segmentation three ways: membrane+DAPI combined, DAPI-only, membrane-only. Results are stored as labeled mask arrays.

**Outputs:** `results/membrane_dapi_segmentation_masks.npy`, `results/dapi_segmentation_masks.npy`, `results/membrane_segmentation_masks.npy`

### Steps 11–12: Masks to GeoDataFrame
Converts segmentation masks to polygon GeoDataFrames and saves as GeoParquet for downstream spatial analysis (e.g., cell-by-gene matrix construction).

**Outputs:** `results/*.parquet`

---

## Output Structure

```
<output_root>/<job_title>/
├── .checkpoints/
│   ├── fixed.tiff                          # Converted fixed image
│   ├── moving.tiff                         # Converted moving image
│   ├── linear_transform.npy                # VALIS affine transform (3×3 inverse map)
│   ├── mirage_transform.npy                # MIRAGE displacement field (H×W×2)
│   ├── recommended_mirage_params.npy       # Auto-recommended MIRAGE hyperparameters
│   ├── high_res_fixed_dapi_filtered_level_0.npy
│   ├── high_res_moving_dapi_filtered_level_0.npy
│   ├── moving_dapi_linear_warped.npy       # DAPI after affine warp (MIRAGE input)
│   ├── pipeline_status.txt                 # Last completed step (for QC viewer)
│   └── valis_output/                       # VALIS intermediate files + overlap images
└── results/
    ├── moving_complete_transform.ome.tiff  # Final warped multi-channel image (pyramidal)
    ├── *_segmentation_masks.npy            # CellPose masks (3 variants)
    ├── *_segmentation_masks.parquet        # Mask polygons as GeoDataFrame
    ├── centroid_quiver_crop_*.png          # Pre-MIRAGE displacement visualization
    ├── centroid_eval_crop_*.png            # Before/after MIRAGE centroid alignment
    ├── centroid_eval_summary.png           # Aggregate alignment improvement plot
    ├── mirage_param_recommendation.png     # Recommended MIRAGE parameter table
    └── qc_report.html                      # Full QC report (generated by `view` command)
```

---

## Requirements

**Python:** 3.8+

**Key dependencies:**
- `valis-wsi` — multi-resolution affine registration
- `tensorflow` — MIRAGE non-linear registration (GPU required)
- `cellpose` — cell segmentation
- `pyvips` / `tifffile` — image I/O and pyramid building
- `opencv-python` — image processing
- `scikit-image`, `scipy`, `numpy` — image analysis utilities
- `geopandas`, `shapely` — spatial mask handling
- `typer`, `loguru` — CLI and logging

**Hardware:** A CUDA-capable GPU is required for MIRAGE (step 6). All other steps run on CPU.

**Conda environment:** The pipeline is tested against the `contamination` conda environment at `/usersoftware/peerd/ghoshr/.conda/envs/contamination/`.
