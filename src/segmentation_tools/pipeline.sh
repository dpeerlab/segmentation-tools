w #!/bin/bash

JOB_TITLE=$1
FIXED_FILE=$2
MOVING_FILE=$3
OUTPUT_ROOT=$4

HIGH_RES_LEVEL=0
FIXED_DAPI_CHANNEL=0
MOVING_DAPI_CHANNEL=1

CHECKPOINTS_DIR=${OUTPUT_ROOT}/${JOB_TITLE}/.checkpoints
RESULTS_DIR=${OUTPUT_ROOT}/${JOB_TITLE}/results

set -euo pipefail
# echo "Processing Job: ${JOB_TITLE}"

# Step 1 - Set up directories
# Idea is that there is output dir is the root and then multiple samples under that
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/000_setup_directories.py \
    --output-root ${OUTPUT_ROOT} \
    --job-title "${JOB_TITLE}"

# echo "Directories set up at ${OUTPUT_ROOT}/${JOB_TITLE}"

# Step 2: Convert to tiff
python \
    "/data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/001_convert_to_tiff.py" \
    --input-path ${FIXED_FILE} \
    --output-root ${OUTPUT_ROOT}/${JOB_TITLE} \
    --prefix "fixed"

# echo "Fixed image converted to TIFF."

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/001_convert_to_tiff.py \
    --input-path ${MOVING_FILE} \
    --output-root ${OUTPUT_ROOT}/${JOB_TITLE} \
    --prefix "moving"

# echo "Moving image converted to TIFF."

# Step 4 - VALIS alignment (replaces steps 002, 003@SIFT, 004)
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/004_valis_alignment.py \
    --fixed-file-path ${CHECKPOINTS_DIR}/fixed.tiff \
    --moving-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --fixed-dapi-channel ${FIXED_DAPI_CHANNEL} \
    --moving-dapi-channel ${MOVING_DAPI_CHANNEL}

echo "VALIS alignment completed."

# Step 4.5 - Preprocess high-res images
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/003_preprocess_images.py \
    --input-file-path ${CHECKPOINTS_DIR}/fixed.tiff \
    --dapi-channel-moving ${FIXED_DAPI_CHANNEL} \
    --level ${HIGH_RES_LEVEL} \
    --output-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --filter

# echo "High-res fixed image preprocessed."

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/003_preprocess_images.py \
    --input-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --dapi-channel-moving ${MOVING_DAPI_CHANNEL} \
    --level ${HIGH_RES_LEVEL} \
    --output-file-path ${CHECKPOINTS_DIR}/high_res_moving_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --filter


# echo "High-res moving image preprocessed."

# # Step 5 - Warp high-res moving image
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/005_warp_image_with_sift.py \
    --moving-file-path ${CHECKPOINTS_DIR}/high_res_moving_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --transform-file-path ${CHECKPOINTS_DIR}/linear_transform.npy

# echo "High-res moving image warped."

# Step 5b - Recommend MIRAGE parameters from tissue crops
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/005b_recommend_mirage_params.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --crop-size 1000 \
    --n-crops 5

echo "MIRAGE parameter recommendation completed."

BATCH_SIZE=1024
LEARNING_RATE=0.012575
NUM_STEPS=2048
# Step 6 - MIRAGE
python /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/006_run_mirage.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --num-steps ${NUM_STEPS}

echo "MIRAGE non-linear registration completed."

# Step 6b - Evaluate MIRAGE alignment via centroid matching
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/006b_evaluate_mirage_alignment.py \
    --warped-file-path ${CHECKPOINTS_DIR}/moving_dapi_linear_warped.npy \
    --fixed-file-path ${CHECKPOINTS_DIR}/high_res_fixed_dapi_filtered_level_${HIGH_RES_LEVEL}.npy \
    --mirage-transform-path ${CHECKPOINTS_DIR}/mirage_transform.npy \
    --crop-size 1000 \
    --n-crops 5

echo "MIRAGE alignment evaluation completed."

# (Optional) - calculate SSIM and plot
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/010_ssim_checks.py \
        ${CHECKPOINTS_DIR}/ \
        || true

# Step 7 - Warp all channels and downsample
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/007_warp_all_channels_and_downsample.py \
    --moving-file-path ${CHECKPOINTS_DIR}/moving.tiff \
    --high-res-level 0

# echo "All channels warped and downsampled."

# Step 8 - Cellpose segmentation Combined DAPI + Membrane
python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/008_segment_with_cellpose.py \
   --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
   --dapi-channel 1 \
   --membrane-channel 0 \

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/008_segment_with_cellpose.py \
   --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
   --dapi-channel 1

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/008_segment_with_cellpose.py \
   --warped-moving-file-path ${RESULTS_DIR}/moving_complete_transform.ome.tiff \
   --membrane-channel 0

# echo "Nuclei Cellpose segmentation completed."

# # Step 10 - Convert masks to GeoDataFrame
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

python \
    /data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/processes/012_combine_combined_and_nuclei_masks.py \
    --combined-masks ${RESULTS_DIR}/membrane_dapi_segmentation_masks.parquet \
    --nuclei-masks ${RESULTS_DIR}/dapi_segmentation_masks.parquet
    