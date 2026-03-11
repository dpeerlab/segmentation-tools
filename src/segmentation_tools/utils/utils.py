
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

from skimage.util import img_as_uint
from loguru import logger
import matplotlib.pyplot as plt

def normalize(
    img: np.ndarray,
    quantiles: list[float] = [0.001, 0.99],
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
            normalize(
                img[..., c],
                quantiles,
                clahe_clip_limit,
                clahe_tile_grid_size,
                return_float,
            )
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

    return img_clahe.astype(np.float32) / 65535.0

def get_multiotsu_threshold(image, n_samples=20, visualize=False):
    """Find the lowest Multi-Otsu threshold by averaging results from sampled patches.

    Tries multi-Otsu with decreasing numbers of classes (4 → 3 → 2) per patch
    so it works regardless of how many distinct intensity modes the tissue has.
    The first threshold separates background from signal in all cases.

    Args:
        image (np.array): 2D float image in [0, 1].
        n_samples (int): Number of valid patches to accumulate before returning.
        visualize (bool): If True, plots the thresholded patches.

    Returns:
        float: Mean of the lowest Otsu threshold across all valid patches.
    """
    H, W = image.shape

    # Glimpse size: 20% of shorter dimension, clamped to [100, 1000]
    glimpse_dim = int(np.clip(int(min(H, W) * 0.2), 100, 1000))
    logger.info(f"Image shape: ({H}, {W}) -> glimpse size: {glimpse_dim}px")

    if visualize:
        rows = int(np.ceil(np.sqrt(n_samples)))
        cols = int(np.ceil(n_samples / rows))
        _, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten()
        plot_index = 0

    valid_thresholds = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(valid_thresholds) < n_samples and attempts < max_attempts:
        attempts += 1

        max_y, max_x = H - glimpse_dim, W - glimpse_dim
        if max_y <= 0 or max_x <= 0:
            glimpse = image
        else:
            y = np.random.randint(0, max_y + 1)
            x = np.random.randint(0, max_x + 1)
            glimpse = image[y : y + glimpse_dim, x : x + glimpse_dim]

        # Skip near-constant patches (background or saturated regions)
        if np.var(glimpse) < 1e-4:
            continue

        # Try multi-Otsu with decreasing class counts until one succeeds.
        # classes=2 is standard Otsu (always succeeds for non-constant images).
        threshold = None
        for n_classes in [4, 3, 2]:
            try:
                thresholds = threshold_multiotsu(glimpse, classes=n_classes)
                t = thresholds[0]
                # Reject if threshold doesn't actually separate anything
                if np.any(glimpse < t) and np.any(glimpse >= t):
                    threshold = t
                    break
            except (RuntimeError, ValueError):
                continue

        if threshold is None:
            continue

        valid_thresholds.append(threshold)

        if visualize and plot_index < len(axes):
            masked = np.where(glimpse < threshold, 0, glimpse)
            axes[plot_index].imshow(masked, cmap="gray")
            axes[plot_index].set_title(f"t={threshold:.3f}")
            axes[plot_index].axis("off")
            plot_index += 1

    if not valid_thresholds:
        raise ValueError(
            "Could not find valid patches to compute a threshold. "
            "Check that the image contains tissue signal."
        )

    if visualize:
        for i in range(plot_index, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    final_threshold = float(np.mean(valid_thresholds))
    logger.info(
        f"Otsu threshold: {final_threshold:.4f} "
        f"(from {len(valid_thresholds)} patches, {attempts} attempts)"
    )
    return final_threshold



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

