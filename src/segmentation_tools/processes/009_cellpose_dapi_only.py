import scipy
import numpy as np
from run_cellpose import models, core, io, plot
import matplotlib.pyplot as plt
from loguru import logger
import sys
from pathlib import Path
import os
import tifffile

def main(warped_moving_file_path = os.PathLike, result_dir = os.PathLike, dapi_channel = int):
    """
    Segments the warped moving image using Cellpose and saves the segmentation masks.
    """
    io.logger_setup()
    logger.info("Nuclei CellPose logger set up")

    if core.use_gpu()==False:
        raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)

    # Load the warped moving image
    warped_moving_image = tifffile.imread(warped_moving_file_path, series=0, level=0, key = dapi_channel)
    logger.info(f"Warped moving image shape: {warped_moving_image.shape}")

    # Select channels for segmentation (e.g., DAPI and membrane)
    dapi_image = warped_moving_image[dapi_channel, :, :]

    # Segment the image
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0

    logger.info("Starting Nuclei CellPose")
    masks, flows, styles = model.eval(
        dapi_image, 
        augment = True,
        batch_size=2,
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        progress=True
    )
    logger.info(f"Nuclei CellPose Finished")

    cell_prob_logit = flows[2]
    cell_prob = scipy.special.expit(cell_prob_logit)
    output_cell_prob_path = Path(result_dir) / "cell_probabilities.npy"
    np.save(output_cell_prob_path, cell_prob)
    logger.info(f"Cell probabilities saved to {output_cell_prob_path}")

    # Save the segmentation masks
    output_mask_path = Path(result_dir) / "nuclei_segmentation_masks.npy"
    np.save(output_mask_path, masks.astype(np.uint32))
    logger.info(f"Segmentation masks saved to {output_mask_path}")
    return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error(
            "Usage: python 006_warp_all_channels_and_downsample.py <warped_file> <dapi_channel>"
        )
        sys.exit(1)

    warped_moving_file_path = sys.argv[1]
    dapi_channel = int(sys.argv[2])
    result_dir = Path(warped_moving_file_path).parent

    main(
        warped_moving_file_path=warped_moving_file_path,
        dapi_channel=dapi_channel,
        result_dir=result_dir
    )