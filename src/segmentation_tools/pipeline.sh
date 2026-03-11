#!/bin/bash
#
# Pipeline steps:
#   1  Setup directories + convert inputs to OME-TIFF
#   2  VALIS rigid/affine alignment  -> linear_transform.npy
#   3  Preprocess high-res DAPI (normalize + Otsu filter)
#   4  Apply linear transform to preprocessed DAPI  -> moving_dapi_linear_warped.npy
#   5  Recommend MIRAGE hyperparameters from tissue crops
#   6  MIRAGE non-linear registration  -> mirage_transform.npy
#   7  Evaluate MIRAGE alignment quality (centroid matching)
#   8  Apply combined transform to all channels  -> moving_complete_transform.ome.tiff
#   9  CellPose segmentation + convert masks to GeoDataFrame parquet

JOB_TITLE=$1
FIXED_FILE=$2
MOVING_FILE=$3
OUTPUT_ROOT=$4
START_STEP=${5:-1}   # First step to run (default: 1 = run all)
END_STEP=${6:-9}     # Last step to run (default: 9 = run all)

HIGH_RES_LEVEL=0
FIXED_DAPI_CHANNEL=0
MOVING_DAPI_CHANNEL=1

CHECKPOINTS_DIR=${OUTPUT_ROOT}/${JOB_TITLE}/.checkpoints
RESULTS_DIR=${OUTPUT_ROOT}/${JOB_TITLE}/results

PROCESSES=/data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes

set -euo pipefail

PIPELINE_START=$(date +%s)

# Helper: run this step only if START_STEP <= step_num <= END_STEP
run_step() {
    local step_num=$1
    [ "${step_num}" -ge "${START_STEP}" ] && [ "${step_num}" -le "${END_STEP}" ]
}

# Print step banner and write status file
progress() {
    local step_num=$1
    local step_name=$2
    local elapsed=$(( $(date +%s) - PIPELINE_START ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [Step ${step_num}/9] ${step_name}  (+${mins}m${secs}s elapsed)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    mkdir -p "${CHECKPOINTS_DIR}"
    echo "${step_num}|${step_name}|$(date -Iseconds)|running" \
        > "${CHECKPOINTS_DIR}/pipeline_status.txt"
}

done_step() {
    local step_num=$1
    local step_name=$2
    local elapsed=$(( $(date +%s) - PIPELINE_START ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo "  ✓ Done  (+${mins}m${secs}s total)"
    echo "${step_num}|${step_name}|$(date -Iseconds)|done" \
        > "${CHECKPOINTS_DIR}/pipeline_status.txt"
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Setup + Convert to TIFF
# ─────────────────────────────────────────────────────────────────────────────
if run_step 1; then
progress 1 "Setup directories + convert to OME-TIFF"
python ${PROCESSES}/000_setup_directories.py \
    --output-root ${OUTPUT_ROOT} \
    --job-title "${JOB_TITLE}"

python ${PROCESSES}/001_convert_to_tiff.py \
    --input-path ${FIXED_FILE} \
    --output-root ${OUTPUT_ROOT}/${JOB_TITLE} \
    --prefix "fixed"

python ${PROCESSES}/001_convert_to_tiff.py \
    --input-path ${MOVING_FILE} \
    --output-root ${OUTPUT_ROOT}/${JOB_TITLE} \
    --prefix "moving"
done_step 1 "Setup + Convert to OME-TIFF"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — VALIS rigid/affine alignment
# ─────────────────────────────────────────────────────────────────────────────
if run_step 2; then
progress 2 "VALIS rigid/affine alignment"
python ${PROCESSES}/004_valis_alignment.py \
    --fixed-file-path     ${CHECKPOINTS_DIR}/fixed.tiff \
    --moving-file-path    ${CHECKPOINTS_DIR}/moving.tiff \
    --fixed-dapi-channel  ${FIXED_DAPI_CHANNEL} \
    --moving-dapi-channel ${MOVING_DAPI_CHANNEL}
done_step 2 "VALIS rigid/affine alignment"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Preprocess high-res DAPI (normalize + Otsu filter)
# ─────────────────────────────────────────────────────────────────────────────
if run_step 3; then
progress 3 "Preprocess high-res DAPI images"
python ${PROCESSES}/003_preprocess_images.py \
    --input-file-path ${CHECKPOINTS_DIR}/fixed.tiff \
    --dapi-channel-moving ${FIXED_DAPI_CHANNEL} \
    --level ${HIGH_RES_LEVEL} \
    --output-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --filter

python ${PROCESSES}/003_preprocess_images.py \
    --input-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --dapi-channel-moving ${MOVING_DAPI_CHANNEL} \
    --level ${HIGH_RES_LEVEL} \
    --output-file-path ${CHECKPOINTS_DIR}/high_res_moving_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --filter
done_step 3 "Preprocess high-res DAPI images"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Apply linear transform to preprocessed DAPI
# ─────────────────────────────────────────────────────────────────────────────
if run_step 4; then
progress 4 "Apply linear transform to moving DAPI"
python ${PROCESSES}/005_warp_image_with_sift.py \
    --moving-file-path   ${CHECKPOINTS_DIR}/high_res_moving_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --transform-file-path ${CHECKPOINTS_DIR}/linear_transform.npy
done_step 4 "Apply linear transform to moving DAPI"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Recommend MIRAGE hyperparameters from tissue crops
# ─────────────────────────────────────────────────────────────────────────────
if run_step 5; then
progress 5 "Recommend MIRAGE hyperparameters"
python ${PROCESSES}/005b_recommend_mirage_params.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path  ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --crop-size 1000 \
    --n-crops 5
done_step 5 "Recommend MIRAGE hyperparameters"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — MIRAGE non-linear registration
# ─────────────────────────────────────────────────────────────────────────────
if run_step 6; then
progress 6 "MIRAGE non-linear registration"
BATCH_SIZE=1024
LEARNING_RATE=0.012575
NUM_STEPS=2048
python ${PROCESSES}/006_run_mirage.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path  ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --batch-size      ${BATCH_SIZE} \
    --learning-rate   ${LEARNING_RATE} \
    --num-steps       ${NUM_STEPS}
done_step 6 "MIRAGE non-linear registration"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Evaluate MIRAGE alignment quality (centroid matching)
# ─────────────────────────────────────────────────────────────────────────────
if run_step 7; then
progress 7 "Evaluate MIRAGE alignment quality"
python ${PROCESSES}/006b_evaluate_mirage_alignment.py \
    --warped-file-path    ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path     ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --mirage-transform-path ${CHECKPOINTS_DIR}/mirage_transform.npy \
    --crop-size 1000 \
    --n-crops 5
done_step 7 "Evaluate MIRAGE alignment quality"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Apply combined transform to all channels + build pyramid OME-TIFF
# ─────────────────────────────────────────────────────────────────────────────
if run_step 8; then
progress 8 "Warp all channels + build pyramid OME-TIFF"
python ${PROCESSES}/007_warp_all_channels_and_downsample.py \
    --moving-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --high-res-level 0
done_step 8 "Warp all channels + build pyramid OME-TIFF"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — CellPose segmentation + convert masks to GeoDataFrame parquet
# ─────────────────────────────────────────────────────────────────────────────
if run_step 9; then
progress 9 "CellPose segmentation + masks to GeoDataFrame"
python ${PROCESSES}/008_segment_with_cellpose.py \
    --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
    --dapi-channel 1 \
    --membrane-channel 0

python ${PROCESSES}/008_segment_with_cellpose.py \
    --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
    --dapi-channel 1

python ${PROCESSES}/008_segment_with_cellpose.py \
    --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
    --membrane-channel 0

python ${PROCESSES}/011_convert_masks_to_gpd.py \
    --masks ${RESULTS_DIR}/membrane_dapi_segmentation_masks.npy \
    --prefix "membrane_dapi"

python ${PROCESSES}/011_convert_masks_to_gpd.py \
    --masks ${RESULTS_DIR}/dapi_segmentation_masks.npy \
    --prefix "dapi"

python ${PROCESSES}/011_convert_masks_to_gpd.py \
    --masks ${RESULTS_DIR}/membrane_segmentation_masks.npy \
    --prefix "membrane"

python ${PROCESSES}/012_combine_combined_and_nuclei_masks.py \
    --combined-masks ${RESULTS_DIR}/membrane_dapi_segmentation_masks.parquet \
    --nuclei-masks   ${RESULTS_DIR}/dapi_segmentation_masks.parquet
done_step 9 "CellPose segmentation + masks to GeoDataFrame"
fi

# ─────────────────────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - PIPELINE_START ))
TOTAL_MINS=$(( TOTAL / 60 ))
TOTAL_SECS=$(( TOTAL % 60 ))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PIPELINE COMPLETE  (${TOTAL_MINS}m${TOTAL_SECS}s total)"
echo "  Results: ${RESULTS_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "complete|all|$(date -Iseconds)|done" \
    > "${CHECKPOINTS_DIR}/pipeline_status.txt"
