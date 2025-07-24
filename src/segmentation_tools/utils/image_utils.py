import os
import matplotlib.pyplot as plt
from pathlib import Path

import skimage
import tifffile
import numpy as np
import cv2
from segmentation_tools.logger import logger
from numpy.typing import ArrayLike
from skimage.util import img_as_uint, img_as_ubyte
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform


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


def get_level_transform(path_to_tiff: os.PathLike, level_to: int, level_from: int = 0) -> AffineTransform:
    """
    Compute the scaling transform (AffineTransform) between two levels of a TIFF pyramid.

    Args:
        path_to_tiff (os.PathLike): Path to the TIFF file.
        level_to (int): Target resolution level.
        level_from (int): Source resolution level (default is 0, i.e. highest res).

    Returns:
        AffineTransform: An affine transform representing the scale between the levels.
    """
    tf = tifffile.TiffFile(path_to_tiff)
    lvl = tf.series[0].levels[level_from]
    from_dims = dict(zip(lvl.axes, lvl.shape))
    lvl = tf.series[0].levels[level_to]
    to_dims = dict(zip(lvl.axes, lvl.shape))
    scale = to_dims["X"] / from_dims["X"], to_dims["Y"] / from_dims["Y"]
    return AffineTransform(scale=scale)

def get_crop_transform(path_to_tiff: os.PathLike, level_to: int, level_from: int = 0) -> AffineTransform:
    """
    Compute the affine transform for cropping between two resolution levels of a TIFF file.

    Args:
        path_to_tiff (os.PathLike): Path to the TIFF file.
        level_to (int): Resolution level to crop to.
        level_from (int): Reference resolution level (default is 0).

    Returns:
        AffineTransform: Affine scaling transform between the levels.
    """
    tf = tifffile.TiffFile(path_to_tiff)
    lvl = tf.series[0].levels[level_from]
    from_dims = dict(zip(lvl.axes, lvl.shape))
    lvl = tf.series[0].levels[level_to]
    to_dims = dict(zip(lvl.axes, lvl.shape))
    scale = to_dims["X"] / from_dims["X"], to_dims["Y"] / from_dims["Y"]
    return AffineTransform(scale=scale)


def get_clip_vals(path_to_tiff: os.PathLike, quantiles: ArrayLike = [0.001, 0.999], **kwargs) -> np.ndarray:
    """
    Compute intensity clipping thresholds from a TIFF image using quantiles.

    Args:
        path_to_tiff (os.PathLike): Path to the TIFF file.
        quantiles (ArrayLike): Quantile range (e.g., [0.001, 0.999]) to clip to.
        **kwargs: Additional keyword arguments passed to tifffile.imread.

    Returns:
        np.ndarray: Array of lower and upper clip values.
    """

    img_ref = tifffile.imread(
        path_to_tiff,
        **kwargs,
    )
    return np.quantile(img_ref, quantiles)


def normalize(
    img: np.ndarray,
    quantiles: ArrayLike = [0.001, 0.999],
    clahe_clip_limit: float = 1.0,
    clahe_tile_grid_size: tuple[int, int] = (20, 20),
    return_float: bool = False,
) -> np.ndarray:
    """
    Normalize an image by clipping intensities to given quantiles and applying CLAHE.

    Args:
        img (np.ndarray): Input image.
        quantiles (ArrayLike): Quantile values for intensity clipping (e.g., [0.001, 0.999]).
        clahe_clip_limit (float): Clip limit for Contrast Limited Adaptive Histogram Equalization.
        clahe_tile_grid_size (tuple[int, int]): Tile grid size for CLAHE.
        return_float (bool): Whether to return output as float32 in [0, 1] or uint8 in [0, 255].

    Returns:
        np.ndarray: Normalized image.
    """

    # 1. Clip intensities to quantiles
    lo, hi = np.quantile(img, quantiles)
    img = np.clip(img, lo, hi)

    # 2. Scale to [0, 1]
    img = (img - lo) / (hi - lo)

    # 3. Convert to uint16 for CLAHE
    img_uint16 = img_as_uint(img)

    # 4. CLAHE in uint16
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
    )
    img_clahe = clahe.apply(img_uint16)

    if return_float:
        return img_clahe.astype(np.float32) / 65535.0  # consistent float [0, 1]
    else:
        return img_as_ubyte(img_clahe / 65535.0)  # match float→uint8 expectations


def save_visualization_overlay(
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    output_file_path: str = "image_moving_warped_overlay.png",
    title: str | None = None,
) -> str:
    """
    Save an overlay of two images with fixed image in red and moving image in cyan.

    Args:
        image_fixed (np.ndarray): The fixed image (e.g., reference or target).
        image_moving (np.ndarray): The moving image to overlay after alignment.
        output_file_path (str): Path to save the resulting overlay image (PNG).
        title (str, optional): Optional plot title.

    Returns:
        str: Path to the saved overlay image.
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_fixed, cmap="Reds", alpha=0.5)
    ax.imshow(image_moving, cmap="Blues", alpha=0.5)

    if title is None:
        title = f"Overlay: Fixed ({image_fixed.shape}) and Warped Moving ({image_moving.shape})"
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(
        f"Overlay saved to: {output_file_path} with fixed shape {image_fixed.shape} and moving shape {image_moving.shape}"
    )
    return output_file_path


def save_visualization(
    image: np.ndarray,
    output_file_path: str,
    title: str | None = None,
) -> str:
    """
    Save a single image visualization with optional title and axis off.

    Args:
        image (np.ndarray): Image to visualize.
        output_file_path (str): Path to save the output image.
        title (str, optional): Title to display on the image.

    Returns:
        str: Path to the saved visualization image.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    if not title:
        title = "Visualization"
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"Image saved to: {output_file_path}")
    return output_file_path


def save_image(
    image: np.ndarray,
    output_file_path: str,
    description: str = "",
) -> None:
    """
    Save an image to a TIFF file with proper type handling and logging.

    Args:
        image (np.ndarray): Image to save. Can be float [0,1], float [0,255], or uint8.
        output_file_path (str): Destination path for the TIFF file.
        description (str): Optional description for logging.

    Returns:
        None
    """

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = img_as_ubyte(np.clip(image, 0, 1))  # safe float-to-uint8 scaling
        else:
            # assume already in 0–255 float, just clip and cast
            image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    tifffile.imwrite(output_file_path, image)
    logger.info(
        f"{description} image saved to: {output_file_path} with shape {image.shape} and dtype {image.dtype}"
    )
