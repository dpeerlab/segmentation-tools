import os
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from numpy.typing import ArrayLike
from skimage.util import img_as_ubyte, img_as_uint
from skimage.exposure import match_histograms

from concurrent.futures import ThreadPoolExecutor, as_completed

from segmentation_tools.logger import logger


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


def create_rgb_overlay(fixed_crop, moving_crop, fixed_on_top=False):
    # Normalize
    fixed = fixed_crop.astype(np.float32)
    moving = moving_crop.astype(np.float32)

    if fixed.max() > 0:
        fixed /= fixed.max()
    if moving.max() > 0:
        moving /= moving.max()

    h, w = fixed.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Always color fixed as red, moving as cyan
    rgb_fixed = np.zeros((h, w, 3), dtype=np.float32)
    rgb_fixed[..., 0] = fixed

    rgb_moving = np.zeros((h, w, 3), dtype=np.float32)
    rgb_moving[..., 1] = moving
    rgb_moving[..., 2] = moving

    # Compose based on order
    if fixed_on_top:
        rgb = rgb_moving * 0.5 + rgb_fixed * 0.5
    else:
        rgb = rgb_fixed * 0.5 + rgb_moving * 0.5

    return rgb

def save_full_overlay(
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    output_file_path: str,
    boxes=None,
    edge_color="lime",
    linewidth=2,
    fixed_on_top: bool = False,
    title: str | None = None,
) -> str:
    """
    Save a full overlay of fixed (red) and moving (cyan) images.
    Draws fixed on top if `fixed_on_top=True`.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if fixed_on_top:
        ax.imshow(image_moving, cmap="Blues", alpha=0.5)
        ax.imshow(image_fixed, cmap="Reds", alpha=0.5)
    else:
        ax.imshow(image_fixed, cmap="Reds", alpha=0.5)
        ax.imshow(image_moving, cmap="Blues", alpha=0.5)

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
        title = f"Overlay: Fixed {image_fixed.shape}, Moving {image_moving.shape}"
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(
        f"Full overlay saved to: {output_file_path} with fixed shape {image_fixed.shape} and moving shape {image_moving.shape}"
    )
    return fig

def save_cropped_overlays(
    boxes,
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    output_file_path: str,
    fixed_on_top: bool = False,
) -> None:
    """
    Save RGB overlays for cropped regions defined by `boxes`.
    Fixed is red, moving is cyan.
    Draw order is controlled by `fixed_on_top`.
    """
    for i, poly in enumerate(boxes):
        minx, miny, maxx, maxy = map(int, poly.bounds)
        crop_fixed = image_fixed[miny:maxy, minx:maxx]
        crop_moving = image_moving[miny:maxy, minx:maxx]
        crop_rgb = create_rgb_overlay(crop_fixed, crop_moving, fixed_on_top)

        fig_crop, ax_crop = plt.subplots(figsize=(4, 4))
        ax_crop.imshow(crop_rgb)
        ax_crop.set_title(
            f"Cropped overlay region #{i} at ({minx},{miny}) to ({maxx},{maxy})"
        )
        ax_crop.axis("off")

        cropped_output_path = (
            f"{output_file_path}_cropped_{int(minx)}_{int(miny)}.png"
        )
        fig_crop.savefig(cropped_output_path, dpi=300, bbox_inches="tight")
        plt.close(fig_crop)

        logger.info(f"Cropped region saved to: {cropped_output_path}")
        return

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


def match_image_histograms(img1, img2):
    def to_uint8_safe(img):
        # If max > 1, assume already in 0–255 scale; otherwise scale up
        if img.max() <= 1.0:
            img = (img * 255.0)
        return np.clip(img, 0, 255).astype(np.uint8)

    m1 = np.mean(img1)
    m2 = np.mean(img2)

    # Histogram match using float32 for better accuracy
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if m1 < m2:
        img1 = match_histograms(img1, img2)
    else:
        img2 = match_histograms(img2, img1)

    # Convert both back to uint8 safely
    img1 = to_uint8_safe(img1)
    img2 = to_uint8_safe(img2)

    return img1, img2

def match_image_histograms_local(img1, img2, tile_fraction=0.1, max_workers=8):
    """
    Match histograms between img1 and img2 using local tiles and ThreadPoolExecutor.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image (same shape as img1).
        tile_fraction (float): Fraction of width/height to use as tile size.
        max_workers (int): Number of parallel threads.

    Returns:
        (np.ndarray, np.ndarray): Locally histogram-matched img1, img2.
    """
    assert img1.shape == img2.shape, "Images must have the same shape"
    h, w = img1.shape

    # Determine tile size
    tile_h = max(64, int(h * tile_fraction))
    tile_w = max(64, int(w * tile_fraction))

    def to_uint8_safe(img):
        if img.max() <= 1.0:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def process_tile(y, x):
        y_end = min(h, y + tile_h)
        x_end = min(w, x + tile_w)

        patch1 = img1[y:y_end, x:x_end].astype(np.float32)
        patch2 = img2[y:y_end, x:x_end].astype(np.float32)

        if patch1.mean() < patch2.mean():
            patch1 = match_histograms(patch1, patch2)
        else:
            patch2 = match_histograms(patch2, patch1)

        return (y, x, to_uint8_safe(patch1), to_uint8_safe(patch2))

    # Allocate output images
    out1 = np.zeros_like(img1, dtype=np.uint8)
    out2 = np.zeros_like(img2, dtype=np.uint8)

    # Schedule and collect tile results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_tile, y, x)
            for y in range(0, h, tile_h)
            for x in range(0, w, tile_w)
        ]

        for fut in as_completed(futures):
            y, x, tile1, tile2 = fut.result()
            out1[y:y + tile1.shape[0], x:x + tile1.shape[1]] = tile1
            out2[y:y + tile2.shape[0], x:x + tile2.shape[1]] = tile2

    return out1, out2