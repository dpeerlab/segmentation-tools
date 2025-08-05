from ctypes import resize
import json
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from numpy.typing import ArrayLike
from skimage.util import img_as_ubyte, img_as_uint
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from shapely.geometry import box as shapely_box

from concurrent.futures import ThreadPoolExecutor, as_completed

from segmentation_tools.logger import logger

def normalize(
    img: np.ndarray,
    quantiles: ArrayLike = [0.001, 0.999],
    clahe_clip_limit: float = 1.0,
    clahe_tile_grid_size: tuple[int, int] = (20, 20),
    return_float: bool = True,
) -> np.ndarray:
    """
    Normalize an image by clipping intensities to given quantiles and applying CLAHE.
    If image is RGB, applies normalization to each channel independently without CLAHE.
    """
    if img.ndim == 3 and img.shape[-1] == 3:
        # Normalize each channel separately, no CLAHE
        norm_channels = [
            normalize(img[..., c], quantiles, clahe_clip_limit, clahe_tile_grid_size, return_float)
            for c in range(3)
        ]
        return np.stack(norm_channels, axis=-1)

    # 1. Clip intensities to quantiles
    lo, hi = np.quantile(img, quantiles)
    img = np.clip(img, lo, hi)

    # 2. Scale to [0, 1]
    if hi - lo < 1e-6:
        img = np.zeros_like(img)
    else:
        img = (img - lo) / (hi - lo)

    # 3. Clean and convert
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0, 1)
    img_uint16 = img_as_uint(img)

    # 4. CLAHE in uint16
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
    )
    img_clahe = clahe.apply(img_uint16)

    if return_float:
        return img_clahe.astype(np.float32) / 65535.0
    else:
        return img_as_ubyte(img_clahe / 65535.0)


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
    rgb[..., 0] = fixed      # R
    rgb[..., 1] = moving     # G  
    rgb[..., 2] = moving     # B

    rgb = np.clip(rgb, 0, 1)  # Ensure values are in [0, 1]
    return rgb

def save_full_overlay(
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
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
    rgb_overlay = create_rgb_overlay(fixed=image_fixed, moving=image_moving)

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
        title = f"Overlay: Fixed {image_fixed.shape}, Moving {image_moving.shape}"
    ax.set_title(title)
    if not plot_axis:
        ax.axis("off")

    fig.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(
        f"Full overlay saved to: {output_file_path} with fixed shape {image_fixed.shape} and moving shape {image_moving.shape}"
    )
    return output_file_path


def save_good_region_control_overlay(
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    boxes: list,  # List of poorly aligned polygons
    output_file_path: str,
    max_attempts: int = 100,
    ssim_threshold: float = 0.8,
    max_width: int = 500,
    max_height: int = 500,
) -> str:
    h, w = image_fixed.shape
    crop_w = min(max_width, w // 8)
    crop_h = min(max_height, h // 8)

    half_w = crop_w // 2
    half_h = crop_h // 2

    bad_regions = [shapely_box(*poly.bounds) for poly in boxes]

    fixed_thresh = image_fixed.max() / 5
    moving_thresh = image_moving.max() / 5

    for _ in range(max_attempts):
        cx = np.random.randint(half_w, w - half_w)
        cy = np.random.randint(half_h, h - half_h)

        minx, maxx = cx - half_w, cx + half_w
        miny, maxy = cy - half_h, cy + half_h

        candidate_box = shapely_box(minx, miny, maxx, maxy)
        if any(b.intersects(candidate_box) for b in bad_regions):
            continue

        crop_fixed = image_fixed[miny:maxy, minx:maxx]
        crop_moving = image_moving[miny:maxy, minx:maxx]

        if crop_fixed[crop_fixed > 0].mean() < fixed_thresh or crop_moving[crop_moving > 0].mean() < moving_thresh:
            continue

        try:
            ssim_score, ssim_map = ssim(
                crop_fixed,
                crop_moving,
                data_range=crop_fixed.max() - crop_fixed.min(),
                full=True
            )
        except ValueError:
            continue

        if ssim_score < ssim_threshold:
            continue

        crop_rgb = create_rgb_overlay(crop_fixed, crop_moving)
        crop_rgb = normalize(crop_rgb, quantiles=[0.001, 0.999], return_float=True)

        # Plot overlay and SSIM map side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(crop_rgb)
        axs[0].set_title(f"Overlay at ({minx}, {miny}) with size ({crop_w}, {crop_h})")
        axs[0].axis("off")

        im = axs[1].imshow(ssim_map, cmap="inferno", vmin=0, vmax=1)
        axs[1].set_title(f"SSIM Map ({ssim_score:.2f})")
        axs[1].axis("off")
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

        path = f"{output_file_path}_control_good_region_{minx}_{miny}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Good control region with SSIM map saved to: {path}")
        return path

    logger.warning("No good region found after max attempts.")
    return ""

def save_poorly_aligned_cropped_overlays(
    boxes,
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    output_file_path: str,
) -> None:
    """
    Save RGB overlays and SSIM maps for cropped regions defined by `boxes`.
    Fixed is red, moving is cyan.
    """
    for poly in boxes:
        minx, miny, maxx, maxy = map(int, poly.bounds)
        width = maxx - minx
        height = maxy - miny

        crop_fixed = image_fixed[miny:maxy, minx:maxx]
        crop_moving = image_moving[miny:maxy, minx:maxx]

        # Compute SSIM map
        try:
            ssim_score, ssim_map = ssim(
                crop_fixed, crop_moving,
                data_range=max(crop_fixed.max() - crop_fixed.min(), 1e-5),
                full=True
            )
        except ValueError:
            logger.warning(f"Skipping region ({minx}, {miny}) due to SSIM error.")
            continue

        # Overlay
        crop_rgb = create_rgb_overlay(fixed=crop_fixed, moving=crop_moving)
        crop_rgb = normalize(crop_rgb, quantiles=[0.001, 0.999], return_float=True)

        # Plot both
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(crop_rgb)
        axes[0].set_title(
            f"Overlay at ({minx}, {miny})\nSize: {width}x{height}, SSIM: {ssim_score:.2f}"
        )
        axes[0].axis("off")

        im = axes[1].imshow(ssim_map, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("SSIM map")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Output path
        path = str(Path(output_file_path).with_suffix("")) + f"_poor_overlay_{minx}_{miny}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved poor overlay region with SSIM map to: {path}")
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
    return


def match_image_histograms(img1, img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)

    # Histogram match using float32 for better accuracy
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if m1 < m2:
        img1 = match_histograms(img1, img2)
    else:
        img2 = match_histograms(img2, img1)

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

def save_pyramidal_tiff_from_high_res(
    image: np.ndarray,
    output_file_path: Path,
    n_levels: int,
    downscale_factor: int = 2,
    description: str = "",
):
    # Normalize to [0, 1] if needed
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= image.max()

    image = np.clip(image, 0, 1)

    # Build pyramid using iterative downsampling
    pyramid = []
    current = image
    for _ in range(n_levels):
        current_uint8 = img_as_ubyte(current)
        pyramid.append(current_uint8)

        new_shape = (
            max(1, current.shape[0] // downscale_factor),
            max(1, current.shape[1] // downscale_factor),
        )

        current = resize(
            current,
            new_shape,
            preserve_range=True,
            anti_aliasing=True,
        )
        current = np.clip(current, 0, 1)

    # Write TIFF with subIFDs
    with tifffile.TiffWriter(output_file_path, bigtiff=True) as tif:
        tif.write(
            data=pyramid[0],
            photometric="minisblack",
            planarconfig="contig",
            subifds=len(pyramid) - 1,
            metadata=None,
        )
        for level in pyramid[1:]:
            tif.write(
                data=level,
                photometric="minisblack",
                planarconfig="contig",
                metadata=None,
            )

    logger.info(f"{description} pyramidal TIFF saved to: {output_file_path}")
    return output_file_path
