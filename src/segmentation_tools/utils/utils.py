
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
    """
    Finds the first Multi-Otsu threshold by averaging results from sampled patches.

    Args:
        image (np.array): The input image (e.g., moving_warped).
        n_samples (int): The required number of valid glimpses to sample.
        visualize (bool): If True, plots the thresholded glimpses.

    Returns:
        float: The averaged first threshold (thresholds[0]).
    """
    H, W = image.shape
    
    # --- 1. Determine Glimpse Size (Heuristic: 20% of min dim, capped at 1000) ---
    min_dim = min(H, W)
    glimpse_dim = np.clip(int(min_dim * 0.2), 100, 1000)

    logger.info(f"Image shape: ({H}, {W}) -> Glimpse size: {glimpse_dim}x{glimpse_dim}")
    
    # --- Plotting Setup (If requested) ---
    if visualize and n_samples > 0:
        rows = int(np.ceil(np.sqrt(n_samples)))
        cols = int(np.ceil(n_samples / rows))
        _, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten()
        plot_index = 0
    
    # --- 2. Sample Glimpses and Calculate Thresholds ---
    valid_thresholds = []
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(valid_thresholds) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Determine random crop coordinates
        max_y, max_x = H - glimpse_dim, W - glimpse_dim
        
        # Handle images smaller than the minimum glimpse size
        if max_y < 0 or max_x < 0:
            glimpse = image
        else:
            y = np.random.randint(0, max_y + 1)
            x = np.random.randint(0, max_x + 1)
            glimpse = image[y : y + glimpse_dim, x : x + glimpse_dim]

        # --- 3. Check for Signal (Variance is more robust than max/min or non-zero count) ---
        # Skip if the patch is near-constant (low variance)
        if np.var(glimpse) < 1e-2: 
            continue
            
        # --- 4. Run Otsu and Store Result ---
        try:
            # Multi-Otsu on the glimpse (classes=4)
            thresholds = threshold_multiotsu(glimpse, classes=4)
            masked_glimpse = np.where(glimpse < thresholds[0], 0, glimpse)
            if np.all(masked_glimpse - glimpse == 0):
                continue

            # thresholds = [skimage.filters.threshold_otsu(glimpse)]
            valid_thresholds.append(thresholds[0])
            
            # --- Visualization ---
            if visualize and plot_index < len(axes):
                ax = axes[plot_index]
                # Binarize using the first threshold
                ax.imshow(masked_glimpse - glimpse, cmap="gray")
                # ax.imshow(np.power(masked_glimpse, 0.1), cmap="gray")
                ax.set_title(f'Threshold: {thresholds[0]:.2f}')
                ax.axis('off')
                plot_index += 1
                
        except (RuntimeError, ValueError):
            # Fails if the glimpse doesn't have enough distinct modes for 4 classes
            continue

    # --- 5. Final Result ---
    if not valid_thresholds:
        raise ValueError(
            "Could not find any valid, multi-modal glimpses to calculate the threshold."
            " Try a smaller 'n_samples' or check your image content."
        )
    
    if visualize:
        # Fill any unused subplots and display
        for i in range(plot_index, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    final_threshold = np.mean(valid_thresholds)
    logger.info(f"\nFound {len(valid_thresholds)} valid thresholds out of {attempts} attempts.")
    
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

