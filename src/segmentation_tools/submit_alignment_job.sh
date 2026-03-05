#!/bin/bash

JOB_TITLE="0033151_Region_1"
FIXED_FILE="/data1/peerd/roses3/basal_ablation/data/raw/xenium/output-XETG00174__0033151__Region_1__20250425__201151/morphology_focus/morphology_focus_0000.ome.tif"
MOVING_FILE="/data1/peerd/roses3/basal_ablation/data/raw/if/nd2_files/20250430_NP_TT_XeniumRun_4_NakCD45AF555/0033151/Region_1.nd2"
OUTPUT_ROOT="/data1/peerd/ghoshr/sam_alignment"


CONDA_ENV="contamination"
PIPELINE_SCRIPT="/data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/pipeline.sh"
# PIPELINE_SCRIPT="/data1/peerd/ghoshr/cellpose_only.sh"
sbatch \
    --job-name="sam_example${JOB_TITLE}" \
    --output="/data1/peerd/ghoshr/segmentation_tools/logs/%x_%j.out" \
    --error="/data1/peerd/ghoshr/segmentation_tools/logs/%x_%j.err" \
    --partition=peerd \
    --time=23:00:00 \
    --mem=500G \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --wrap="source ~/.bashrc && conda activate $CONDA_ENV && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH &&  ${PIPELINE_SCRIPT} ${JOB_TITLE} ${FIXED_FILE} ${MOVING_FILE} ${OUTPUT_ROOT}"