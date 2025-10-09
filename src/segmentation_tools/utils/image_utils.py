from pprint import pprint
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from skimage.util import img_as_ubyte, img_as_uint
from skimage.exposure import match_histograms, equalize_adapthist, rescale_intensity
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from shapely.geometry import box as shapely_box
from numpy.typing import ArrayLike

from skimage.util import img_as_ubyte, img_as_uint
import cv2

from concurrent.futures import ThreadPoolExecutor, as_completed

from segmentation_tools.logger import logger

def normalize(
        img: np.ndarray,
        quantiles: list = [0.001, 0.999],
) -> np.ndarray:
    # 1. Quantile Clipping and Scaling (Non-Redundant Step)
    # This cleans up outliers before CLAHE is applied.
    lo, hi = np.quantile(img, quantiles)
    img_clipped = np.clip(img, lo, hi)
    
    # Scale to [0, 1] float, which is ideal input for skimage CLAHE
    if hi - lo < 1e-6:
        img_normalized = np.zeros_like(img_clipped, dtype=np.float32)
    else:
        img_normalized = (img_clipped - lo) / (hi - lo)
        
    img_normalized = np.nan_to_num(img_normalized)
    if img_normalized.max() > 255.0:
        img_normalized /= 65535.0
    elif img_normalized.max() > 1.0:
        img_normalized /= 255.0

    img_normalized = np.clip(img_normalized, 0, 1).astype(np.float32)
    return img_normalized


# def normalize(
#     img: np.ndarray,
#     quantiles: list = [0.001, 0.999],
#     clahe_clip_limit: float = 1.0,
#     clahe_tile_grid_size: tuple[int, int] = (20, 20),
#     return_float: bool = True,
# ) -> np.ndarray:

#     """
#     Normalize an image by clipping intensities to given quantiles and applying CLAHE.
#     If image is RGB, applies normalization to each channel independently without CLAHE.
#     """
#     if img.ndim == 3 and img.shape[-1] == 3:
#         # Normalize each channel separately, no CLAHE
#         norm_channels = [
#             normalize(
#                 img[..., c],
#                 quantiles,
#                 clahe_clip_limit,
#                 clahe_tile_grid_size,
#                 return_float,
#             )
#             for c in range(3)
#         ]
#         return np.stack(norm_channels, axis=-1)

#     # 1. Clip intensities to quantiles
#     lo, hi = np.quantile(img, quantiles)
#     img = np.clip(img, lo, hi)

#     # 2. Scale to [0, 1]
#     if hi - lo < 1e-6:
#         img = np.zeros_like(img)
#     else:
#         img = (img - lo) / (hi - lo)

#     # 3. Clean and convert
#     img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
#     img = np.clip(img, 0, 1)
#     img_uint16 = img_as_uint(img)

#     # 4. CLAHE in uint16
#     clahe = cv2.createCLAHE(
#         clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
#     )
#     img_clahe = clahe.apply(img_uint16)

#     if return_float:
#         return img_clahe.astype(np.float32) / 65535.0
#     else:
#         return img_as_ubyte(img_clahe / 65535.0)


# def normalize(
#     img: np.ndarray,
#     clahe_clip_limit: float = 0.01,          # skimage uses [0,1], not OpenCV’s ~1-40
#     clahe_tile_grid_size: tuple[int, int] = (20, 20),
#     return_float: bool = True,
# ) -> np.ndarray:
#     """
#     Normalize an image using adaptive histogram equalization (CLAHE) with skimage.
#     If image is RGB, applies normalization to each channel independently.
#     """
#     if img.ndim == 3 and img.shape[-1] == 3:
#         logger.info("Shouldn't be here?")
#         # Apply per-channel
#         norm_channels = [
#             normalize(
#                 img[..., c],
#                 clahe_clip_limit,
#                 clahe_tile_grid_size,
#                 return_float,
#             )
#             for c in range(3)
#         ]
#         return np.stack(norm_channels, axis=-1)

#     # Ensure float in [0,1]
#     img = img_as_float(img)
#     img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
#     img = np.clip(img, 0, 1)

#     # Apply skimage CLAHE
#     img_eq = equalize_adapthist(
#         img,
#         clip_limit=clahe_clip_limit,
#         kernel_size=clahe_tile_grid_size
#     )

#     if return_float:
#         return img_eq.astype(np.float32)
#     else:
#         return (img_eq * 255).astype(np.uint8)


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

        if (
            crop_fixed[crop_fixed > 0].mean() < fixed_thresh
            or crop_moving[crop_moving > 0].mean() < moving_thresh
        ):
            continue

        try:
            ssim_score, ssim_map = ssim(
                crop_fixed,
                crop_moving,
                data_range=crop_fixed.max() - crop_fixed.min(),
                full=True,
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
                crop_fixed,
                crop_moving,
                data_range=max(crop_fixed.max() - crop_fixed.min(), 1e-5),
                full=True,
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
        path = (
            str(Path(output_file_path).with_suffix(""))
            + f"_poor_overlay_{minx}_{miny}.png"
        )
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
    Saves an image to a TIFF file, preserving fidelity by handling
    different data ranges and types correctly.

    Args:
        image (np.ndarray): The image array. Expected to be in one of the
                            following ranges: [0, 1], [0, 255], or [0, 65535].
        output_file_path (str): Destination path for the TIFF file.
        description (str): Optional description for logging.
    """
    processed_image = image

    # Handle floating-point data by scaling to the appropriate integer range
    if np.issubdtype(image.dtype, np.floating):
        image_max = image.max()
        
        # Case 1: Normalized float [0, 1]
        if image_max <= 1.0:
            # Scale to 16-bit (uint16) to maximize precision
            # This is safer than scaling to 8-bit, as it preserves more information
            processed_image = np.clip(image, 0, 1) * 65535
            processed_image = processed_image.astype(np.uint16)

        # Case 2: Float in [0, 255] or [0, 65535] range
        elif image_max <= 255.0:
            processed_image = np.clip(image, 0, 255).astype(np.uint8)
        else: # Assumes float in [0, 65535] or larger range
            processed_image = np.clip(image, 0, 65535).astype(np.uint16)

    # For integer data, no scaling is needed, as tifffile handles it correctly
    # The image is saved with its original integer dtype (e.g., uint8, uint16)

    # Save the processed image using tifffile
    tifffile.imwrite(output_file_path, processed_image)
    
    logger.info(
        f"{description} image saved to: {output_file_path} "
        f"with shape {processed_image.shape} and dtype {processed_image.dtype}"
    )
    return

def calculate_shannon_entropy(image):
    """
    Calculates the Shannon entropy of an image's histogram.
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def match_moving_to_fixed(moving, fixed, background_cutoff = 0.15):
    moving = np.where(moving < background_cutoff, 0, moving)
    fixed = np.where(fixed < background_cutoff, 0, fixed)

    moving = moving.astype(np.float32)
    fixed = fixed.astype(np.float32) 
    moving = match_histograms(moving, fixed)

    moving = equalize_adapthist(moving, clip_limit=0.01, kernel_size=(20, 20))
    moving = np.clip(moving, 0, 1)

    fixed = equalize_adapthist(fixed, clip_limit=0.01, kernel_size=(20, 20))
    fixed = np.clip(fixed, 0, 1)

    return moving, fixed


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
            out1[y : y + tile1.shape[0], x : x + tile1.shape[1]] = tile1
            out2[y : y + tile2.shape[0], x : x + tile2.shape[1]] = tile2

    return out1, out2
def save_pyramidal_tiff_multi_channel(
    image_stacked: np.ndarray,
    output_file_path: Path,
    n_levels: int,
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

    for _ in range(n_levels):
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
