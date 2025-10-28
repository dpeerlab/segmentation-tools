import numpy as np
from cellpose import models, core, io, plot
import matplotlib.pyplot as plt
from loguru import logger
import sys
from pathlib import Path
import os
import tifffile

def main(warped_moving_file_path = os.PathLike, result_dir = os.PathLike, dapi_channel = int, membrane_channel = int):
    """
    Segments the warped moving image using Cellpose and saves the segmentation masks.
    """
    io.logger_setup()

    if core.use_gpu()==False:
        raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)

    # Load the warped moving image
    warped_moving_image = tifffile.imread(warped_moving_file_path, series=0, level=0)
    logger.info(f"Warped moving image shape: {warped_moving_image.shape}")

    # Select channels for segmentation (e.g., DAPI and membrane)
    dapi_channel = warped_moving_image[dapi_channel, :, :]
    membrane_channel = warped_moving_image[membrane_channel, :, :]
    img_selected_channels = np.stack([membrane_channel, dapi_channel], axis=0)

    # Segment the image
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0

    masks, flows, styles = model.eval(
        img_selected_channels, 
        augment = True,
        batch_size=4,
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        progress=True
    )

    # Save the segmentation masks
    output_mask_path = Path(result_dir) / "segmentation_masks.tiff"
    tifffile.imwrite(output_mask_path, masks.astype(np.uint32))
    logger.info(f"Segmentation masks saved to {output_mask_path}")
    return

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error(
            "Usage: python 006_warp_all_channels_and_downsample.py <warped_file> <dapi_channel> <membrane_channel>"
        )
        sys.exit(1)

    warped_moving_file_path = sys.argv[1]
    dapi_channel = int(sys.argv[2])
    membrane_channel = int(sys.argv[3])
    result_dir = Path(warped_moving_file_path).parent

    main(
        warped_moving_file_path=warped_moving_file_path,
        dapi_channel=dapi_channel,
        membrane_channel=membrane_channel,
        result_dir=result_dir
    )