
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

from skimage.util import img_as_uint
from loguru import logger

def normalize(
    img: np.ndarray,
    quantiles: list = [0.001, 0.99],
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

def get_multiotsu_threshold(image: np.ndarray, num_classes: int = 4) -> float:
    """
    Computes Otsu's threshold for the given image.
    """
    if np.all(image == image.flat[0]):
        return None

    thresh = threshold_multiotsu(image, classes=num_classes)[0]

    if thresh < 0.05:
        thresh = threshold_multiotsu(image, classes=num_classes-1)[0]
    elif thresh > 0.2:
        thresh = threshold_multiotsu(image, classes=num_classes+1)[0]
    return thresh
