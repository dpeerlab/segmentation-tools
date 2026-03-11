#!/bin/bash

segmentation-tools run \
  -j "0033151_Region_1" \
  -f "/data1/peerd/roses3/basal_ablation/data/raw/xenium/output-XETG00174__0033151__Region_1__20250425__201151/morphology_focus/morphology_focus_0000.ome.tif" \
  -m "/data1/peerd/roses3/basal_ablation/data/raw/if/nd2_files/20250430_NP_TT_XeniumRun_4_NakCD45AF555/0033151/Region_1.nd2" \
  -o "/data1/peerd/ghoshr/sam_alignment" \
  --start-step 3