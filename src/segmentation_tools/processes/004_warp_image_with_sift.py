import sys
from segmentation_tools import logger
import numpy as np
from pathlib import Path
import cupy as cp
import cucim


def main(moving_file_path, transform_file_path, checkpoint_dir):
    moving_image = cp.load(moving_file_path)
    transform_params = cp.load(transform_file_path)
    image_warped = cucim.skimage.transform.warp(
        moving_image.astype("float32"),
        cp.array([transform_params[:, :, 0], transform_params[:, :, 1]]),
        order=4,
    )
    warped_file_path = checkpoint_dir / f"moving_warped.npy"
    cp.save(warped_file_path, image_warped)
    logger.info(
        f"Warped image saved to {warped_file_path}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error(
            "Usage: python 004_warp_image_with_sift.py <moving_file> <transform_file>"
        )
        sys.exit(1)

    moving_file_path = sys.argv[1]
    transform_file_path = sys.argv[2]
    checkpoint_dir = Path(moving_file_path).parent
    main(moving_file_path=moving_file_path, transform_file_path=transform_file_path, checkpoint_dir=checkpoint_dir)
