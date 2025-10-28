import sys
from loguru import logger
import numpy as np
from pathlib import Path
import skimage
import tifffile


def get_shape_at_level(
    tiff_path: Path, level: int = 0, series: int = 0
) -> tuple[int, int]:
    """
    Get the shape (height, width) of a given resolution level without loading the image.

    Args:
        tiff_path: Path to the TIFF file.
        level: Resolution level to query (0 = highest).
        series: Series index to use.

    Returns:
        Tuple of (height, width) for the specified level.
    """
    with tifffile.TiffFile(tiff_path) as tf:
        page = tf.series[series].levels[level]
        shape = page.shape

    # If shape includes channel, strip it
    if len(shape) == 3:
        shape = shape[1], shape[2]

    return shape


def main(moving_file_path, transform_file_path, checkpoint_dir, fixed_shape):
    moving_image = np.load(moving_file_path)
    transform_params = np.load(transform_file_path)
    image_warped = skimage.transform.warp(
        moving_image.astype("float32"),
        transform_params,
        order=0,
        output_shape=fixed_shape,
        mode="constant",
    )
    warped_file_path = checkpoint_dir / f"moving_dapi_warped.npy"
    np.save(warped_file_path, image_warped)
    logger.info(f"Warped image saved to {warped_file_path}")
    return 0


if __name__ == "__main__":
    logger.info("here")
    if len(sys.argv) != 3:
        logger.error(
            "Usage: python 004_warp_image_with_sift.py <moving_file> <transform_file>"
        )
        sys.exit(1)

    moving_file_path = sys.argv[1]
    transform_file_path = sys.argv[2]
    original_fixed_file_path = Path(moving_file_path).parent / "fixed.tiff"
    fixed_shape = get_shape_at_level(original_fixed_file_path, level=0)
    checkpoint_dir = Path(moving_file_path).parent

    main(
        moving_file_path=moving_file_path,
        transform_file_path=transform_file_path,
        checkpoint_dir=checkpoint_dir,
        fixed_shape=fixed_shape,
    )
