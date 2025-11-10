import scipy
import numpy as np
from cellpose import models, core, io, plot
import matplotlib.pyplot as plt
from loguru import logger
import sys
from pathlib import Path
import os
import tifffile
import argparse

def main(warped_moving_file_path: os.PathLike, result_dir: os.PathLike, dapi_channel: int, membrane_channel: int = None):
    """
    Segments the warped moving image using Cellpose and saves the segmentation masks.
    """

    # Load the warped moving image
    warped_moving_image = tifffile.imread(warped_moving_file_path, series=0, level=0)
    logger.info(f"Warped moving image shape: {warped_moving_image.shape}")

    # Select channels for segmentation (e.g., DAPI and membrane)
    dapi_image = warped_moving_image[dapi_channel, :, :]

    if membrane_channel is not None:
        membrane_image = warped_moving_image[membrane_channel, :, :]
        img_selected_channels = np.stack([membrane_image, dapi_image], axis=0)
        prefix = "membrane_dapi"
    else:
        img_selected_channels = dapi_image
        prefix = "dapi"
    
    output_cell_prob_path = Path(result_dir) / f"{prefix}_cell_probabilities.npy"
    output_mask_path = Path(result_dir) / f"{prefix}_segmentation_masks.npy"

    if output_mask_path.exists() and output_cell_prob_path.exists():
        logger.info(f"Segmentation outputs already exist at {output_mask_path} and {output_cell_prob_path}. Skipping computation.")
        return 0

    io.logger_setup()
    logger.info("CellPose logger set up")

    if core.use_gpu()==False:
        raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)

    # Segment the image
    flow_threshold = 0.0
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0

    logger.info("Starting CellPose")
    masks, flows, _ = model.eval(
        img_selected_channels, 
        augment = True,
        batch_size=2,
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        progress=True
    )
    logger.info(f"CellPose Finished")

    cell_prob_logit = flows[2]
    cell_prob = scipy.special.expit(cell_prob_logit)
    
    np.save(output_cell_prob_path, cell_prob)
    logger.info(f"Cell probabilities saved to {output_cell_prob_path}")

    # Save the segmentation masks
    
    np.save(output_mask_path, masks.astype(np.uint32))
    logger.info(f"Segmentation masks saved to {output_mask_path}")
    return

def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Segment warped moving image using CellPose."
    )

    parser.add_argument(
        "--warped-moving-file-path",
        required=True,
        type=str,
        help="Path to the warped moving image OME-TIFF file.",
    )
    parser.add_argument(
        "--dapi-channel",
        required=True,
        type=int,
        help="Index of the DAPI channel in the image.",
    )
    parser.add_argument(
        "--membrane-channel",
        required=False,
        type=int,
        help="Index of the membrane channel in the image.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    warped_moving_file_path = args.warped_moving_file_path
    dapi_channel = args.dapi_channel
    membrane_channel = args.membrane_channel

    logger.info(f"DAPI channel provided: {dapi_channel}")
    
    if membrane_channel is None:
        logger.info("No membrane channel provided, proceeding with DAPI channel only.")
    else:
        logger.info(f"Membrane channel provided: {membrane_channel}")

    result_dir = Path(warped_moving_file_path).parent

    main(
        warped_moving_file_path=warped_moving_file_path,
        dapi_channel=dapi_channel,
        membrane_channel=membrane_channel,
        result_dir=result_dir
    )