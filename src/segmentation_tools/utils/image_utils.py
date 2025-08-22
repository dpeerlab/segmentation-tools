from pprint import pprint
from pathlib import Path

import cucim.skimage as cs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from skimage.util import img_as_ubyte, img_as_uint
from skimage.exposure import match_histograms, equalize_adapthist, rescale_intensity
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from shapely.geometry import box as shapely_box
import cupy as cp

from concurrent.futures import ThreadPoolExecutor, as_completed

from segmentation_tools.logger import logger


def normalize(
    img: cp.ndarray,
    quantiles: list = [0.001, 0.999],
    clahe_clip_limit: float = 0.01,
    clahe_tile_grid_size: tuple[int, int] = None,
) -> cp.ndarray:
    """
    Normalize an image by:
    1. Clipping intensities to given quantiles
    2. Scaling to [0, 1]
    3. Applying CLAHE (adaptive histogram equalization) via CuCIM
    Works for grayscale and multi-channel images.
    """
    def normalize_channel(channel):
        # 1. Clip to robust quantile range
        lo, hi = cp.quantile(channel, quantiles)
        channel = cp.clip(channel, lo, hi)

        # 2. Normalize to [0, 1]
        if hi - lo < 1e-6:
            channel = cp.zeros_like(channel, dtype=cp.float32)
        else:
            channel = (channel - lo) / (hi - lo)
        channel = cp.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=0.0)
        channel = cp.clip(channel, 0.0, 1.0)

        # 3. CLAHE using CuCIM
        channel_eq = cs.exposure.equalize_adapthist(
            channel,
            clip_limit=clahe_clip_limit,
            kernel_size=clahe_tile_grid_size,
        )
        return channel_eq.astype(cp.float32)

    # If RGB or multi-channel, normalize each channel independently
    if img.ndim == 3:
        channels = [
            normalize_channel(img[..., c]) for c in range(img.shape[-1])
        ]
        normalized_image = cp.stack(channels, axis=-1)
    else:
        normalized_image = normalize_channel(img)

    return normalized_image

def create_rgb_overlay(fixed, moving):
    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)

    # Normalize to [0,1] range
    if fixed.max() > 0:
        fixed /= fixed.max()
    if moving.max() > 0:
        moving /= moving.max()

    # Create RGB without perceptual scaling
    rgb = np.zeros((*fixed.shape, 3), dtype=np.float32)
    rgb[..., 0] = fixed  # R
    rgb[..., 1] = moving  # G
    rgb[..., 2] = moving  # B

    rgb = np.clip(rgb, 0, 1)  # Ensure values are in [0, 1]
    return rgb


def save_full_overlay(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    output_file_path: str,
    boxes=None,
    edge_color="lime",
    linewidth=2,
    title: str | None = None,
    plot_axis: bool = False,
) -> str:
    """
    Save an RGB overlay of fixed (red) and moving (cyan) images with optional bounding boxes.
    """
    # Create RGB overlay
    rgb_overlay = create_rgb_overlay(fixed=fixed_image, moving=moving_image)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_overlay)

    if boxes is not None:
        for poly in boxes:
            minx, miny, maxx, maxy = map(int, poly.bounds)
            rect = mpatches.Rectangle(
                (minx, miny),
                maxx - minx,
                maxy - miny,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth,
            )
            ax.add_patch(rect)

    if title is None:
        title = f"Overlay: Fixed {fixed_image.shape}, Moving {moving_image.shape}"
    ax.set_title(title)
    if not plot_axis:
        ax.axis("off")

    fig.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(
        f"Full overlay saved to: {output_file_path} with fixed shape {fixed_image.shape} and moving shape {moving_image.shape}"
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
    Save an image to a TIFF file as uint16, regardless of input type.

    Args:
        image (np.ndarray): Input image. Can be float [0,1], float [0,65535], uint8, etc.
        output_file_path (str): Destination path for the TIFF file.
        description (str): Optional description for logging.

    Returns:
        None
    """
    original_dtype = image.dtype

    # Convert float or uint8 to uint16 safely
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = img_as_uint(image)

    elif image.dtype == np.uint8:
        image = (image.astype(np.uint16) << 8)  # 8-bit to 16-bit scaling

    elif image.dtype != np.uint16:
        image = image.astype(np.uint16)

    tifffile.imwrite(output_file_path, image)
    logger.info(
        f"{description} image saved to: {output_file_path} "
        f"(original dtype: {original_dtype}, saved as uint16, shape: {image.shape})"
    )


def save_pyramidal_tiff_multi_channel(
    image_stacked: np.ndarray,
    output_file_path: Path,
    num_levels: int,
    downscale_factor: int = 2,
    description: str = "",
):

    if image_stacked.ndim == 2:
        image_stacked = image_stacked[np.newaxis, ...]

    # Normalize all channels to [0, 1]
    image_stacked = image_stacked.astype(np.float32)

    if image_stacked.max() > 1:
        image_stacked /= image_stacked.max()
    image_stacked = np.clip(image_stacked, 0, 1)

    pyramid = []
    current = image_stacked.copy()

    for _ in range(num_levels):
        # Convert to uint8 for saving
        current_uint8 = np.stack([img_as_ubyte(ch) for ch in current], axis=0)
        pyramid.append(current_uint8)
        # Downsample for next level
        new_h = max(1, current.shape[1] // downscale_factor)
        new_w = max(1, current.shape[2] // downscale_factor)
        current = np.stack(
            [
                resize(ch, (new_h, new_w), preserve_range=True, anti_aliasing=True)
                for ch in current
            ],
            axis=0,
        )
        if np.isnan(current).any():
            logger.warning("NaNs detected in pyramid level.")
            current = np.nan_to_num(current, nan=0.0)
        current = np.clip(current, 0, 1)

    # Open TiffWriter to write an OME-TIFF (use bigtiff for large pyramids)
    with tifffile.TiffWriter(output_file_path, bigtiff=True) as tif:
        # OME metadata: define axes and channel names
        metadata = {
            'axes': 'CYX',                             # Indicates [Channel, Y, X] axes order
        }
        # Write the base level (level 0) with subIFDs for the other levels
        tif.write(
            pyramid[0], 
            subifds=len(pyramid)-1,  # reserve SubIFDs for levels 1 and 2
            photometric='minisblack',  # grayscale photometric interpretation
            metadata=metadata, 
            dtype=np.uint8
        )
        # Write each downsampled pyramid level as a SubIFD of the previous level
        for level_data in pyramid[1:]:
            tif.write(
                level_data, 
                subfiletype=1,       # mark this image as a reduced-resolution (pyramid) layer
                photometric='minisblack',
                dtype=np.uint8
            )

    logger.info(
        f"{description} multi-channel pyramidal TIFF saved to: {output_file_path}"
    )
    return


def get_num_channels(tiff_file: str, series: int, level) -> int:
    """
    Get the number of channels in a TIFF file.

    Args:
        tiff_file (str): Path to the TIFF file.

    Returns:
        int: Number of channels in the TIFF file.
    """
    with tifffile.TiffFile(tiff_file) as tif:
        return len(tif.series[series].levels[level].pages)

def get_num_levels(tiff_file: str, series: int, level) -> int:
    """
    Get the number of levels in a TIFF file.

    Args:
        tiff_file (str): Path to the TIFF file.

    Returns:
        int: Number of levels in the TIFF file.
    """
    with tifffile.TiffFile(tiff_file) as tif:
        num_levels = (
            len(tif.series[series].levels) - level
        )
    return num_levels
