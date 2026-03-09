#!/bin/bash

JOB_TITLE=$1
FIXED_FILE=$2
MOVING_FILE=$3
OUTPUT_ROOT=$4
START_STEP=${5:-0}   # Optional 5th arg: step number to resume from (default: 0 = run all)

HIGH_RES_LEVEL=0
FIXED_DAPI_CHANNEL=0
MOVING_DAPI_CHANNEL=1

CHECKPOINTS_DIR=${OUTPUT_ROOT}/${JOB_TITLE}/.checkpoints
RESULTS_DIR=${OUTPUT_ROOT}/${JOB_TITLE}/results

set -euo pipefail

PIPELINE_START=$(date +%s)

# Helper: skip a step if START_STEP is set higher than this step's number
run_step() {
    local step_num=$1
    [ "${step_num}" -ge "${START_STEP}" ]
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
    echo "  [Step ${step_num}] ${step_name}  (+${mins}m${secs}s elapsed)"
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

# Step 0 - Set up directories
if run_step 0; then
progress 0 "Setup directories"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/000_setup_directories.py \
    --output-root ${OUTPUT_ROOT} \
    --job-title "${JOB_TITLE}"
done_step 0 "Setup directories"
fi

# Step 1 - Convert to tiff
if run_step 1; then
progress 1 "Convert to TIFF"
python \
    "/data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/001_convert_to_tiff.py" \
    --input-path ${FIXED_FILE} \
    --output-root ${OUTPUT_ROOT}/${JOB_TITLE} \
    --prefix "fixed"

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/001_convert_to_tiff.py \
    --input-path ${MOVING_FILE} \
    --output-root ${OUTPUT_ROOT}/${JOB_TITLE} \
    --prefix "moving"
done_step 1 "Convert to TIFF"
fi

# Step 4 - VALIS alignment
if run_step 4; then
progress 4 "VALIS alignment (rigid/affine)"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/004_valis_alignment.py \
    --fixed-file-path ${CHECKPOINTS_DIR}/fixed.tiff \
    --moving-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --fixed-dapi-channel ${FIXED_DAPI_CHANNEL} \
    --moving-dapi-channel ${MOVING_DAPI_CHANNEL}
done_step 4 "VALIS alignment (rigid/affine)"
fi

# Step 3 - Preprocess high-res images (for MIRAGE)
if run_step 3; then
progress 3 "Preprocess high-res DAPI images"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/003_preprocess_images.py \
    --input-file-path ${CHECKPOINTS_DIR}/fixed.tiff \
    --dapi-channel-moving ${FIXED_DAPI_CHANNEL} \
    --level ${HIGH_RES_LEVEL} \
    --output-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --filter

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/003_preprocess_images.py \
    --input-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --dapi-channel-moving ${MOVING_DAPI_CHANNEL} \
    --level ${HIGH_RES_LEVEL} \
    --output-file-path ${CHECKPOINTS_DIR}/high_res_moving_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --filter
done_step 3 "Preprocess high-res DAPI images"
fi

# Step 5 - Warp high-res moving image with linear transform
if run_step 5; then
progress 5 "Warp moving DAPI with linear transform"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/005_warp_image_with_sift.py \
    --moving-file-path ${CHECKPOINTS_DIR}/high_res_moving_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --transform-file-path ${CHECKPOINTS_DIR}/linear_transform.npy
done_step 5 "Warp moving DAPI with linear transform"
fi

# Step 5b - Recommend MIRAGE parameters from tissue crops
if run_step 51; then
progress 51 "Recommend MIRAGE parameters"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/005b_recommend_mirage_params.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --crop-size 1000 \
    --n-crops 5
done_step 51 "Recommend MIRAGE parameters"
fi

# Step 6 - MIRAGE non-linear registration
if run_step 6; then
progress 6 "MIRAGE non-linear registration (~20-40 min)"
BATCH_SIZE=1024
LEARNING_RATE=0.012575
NUM_STEPS=2048
python /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/006_run_mirage.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --num-steps ${NUM_STEPS}
done_step 6 "MIRAGE non-linear registration"
fi

# Step 6b - Evaluate MIRAGE alignment via centroid matching
if run_step 61; then
progress 61 "Evaluate MIRAGE alignment quality"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/006b_evaluate_mirage_alignment.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --mirage-transform-path ${CHECKPOINTS_DIR}/mirage_transform.npy \
    --crop-size 1000 \
    --n-crops 5
done_step 61 "Evaluate MIRAGE alignment quality"
fi

# Step 10 - SSIM checks (optional, never fails)
if run_step 10; then
progress 10 "SSIM quality checks"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/010_ssim_checks.py \
        ${CHECKPOINTS_DIR}/ \
        || true
done_step 10 "SSIM quality checks"
fi

# Step 7 - Warp all channels and downsample
if run_step 7; then
progress 7 "Warp all channels + build pyramid OME-TIFF"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/007_warp_all_channels_and_downsample.py \
    --moving-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --high-res-level 0
done_step 7 "Warp all channels + build pyramid OME-TIFF"
fi

# Step 8 - Cellpose segmentation
if run_step 8; then
progress 8 "CellPose segmentation (combined, DAPI-only, membrane-only)"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/008_segment_with_cellpose.py \
   --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
   --dapi-channel 1 \
   --membrane-channel 0

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/008_segment_with_cellpose.py \
   --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
   --dapi-channel 1

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/008_segment_with_cellpose.py \
   --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
   --membrane-channel 0
done_step 8 "CellPose segmentation"
fi

# Step 11 - Convert masks to GeoDataFrame
if run_step 11; then
progress 11 "Convert masks to GeoDataFrame (parquet)"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/011_convert_masks_to_gpd.py \
    --masks ${RESULTS_DIR}/membrane_dapi_segmentation_masks.npy \
    --prefix "membrane_dapi"

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/011_convert_masks_to_gpd.py \
    --masks ${RESULTS_DIR}/dapi_segmentation_masks.npy \
    --prefix "dapi"

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/011_convert_masks_to_gpd.py \
    --masks ${RESULTS_DIR}/membrane_segmentation_masks.npy \
    --prefix "membrane"
done_step 11 "Convert masks to GeoDataFrame"
fi

# Step 12 - Combine masks
if run_step 12; then
progress 12 "Combine membrane+DAPI and nuclei masks"
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/012_combine_combined_and_nuclei_masks.py \
    --combined-masks ${RESULTS_DIR}/membrane_dapi_segmentation_masks.parquet \
    --nuclei-masks ${RESULTS_DIR}/dapi_segmentation_masks.parquet
done_step 12 "Combine membrane+DAPI and nuclei masks"
fi

# Pipeline complete
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
