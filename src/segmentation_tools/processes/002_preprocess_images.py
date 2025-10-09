from pathlib import Path
from segmentation_tools.logger import logger
import sys
import numpy as np
import tifffile

from skimage.util import img_as_ubyte, img_as_uint
import cv2
import zarr


def normalize(
    img: np.ndarray,
    quantiles: list = [0.001, 0.999],
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

    if img_clahe.max() > 255.0:
        return img_clahe.astype(np.float32) / 65535.0
    elif img_clahe.max() > 1.0:
        return img_as_ubyte(img_clahe / 255.0)
    else:
        return img_clahe.astype(np.float32)


def load_image(input_file_path: Path, level: int):
    """Load moving and fixed images from a multi-resolution OME-TIFF file."""
    with tifffile.TiffFile(input_file_path) as tif:
        if level >= len(tif.series[0].levels):
            raise ValueError(
                f"Requested level {level} exceeds available levels {len(tif.series[0].levels)-1}"
            )

    image = tifffile.imread(
        input_file_path,
        key=0,
        series=0,
        level=level,
        maxworkers=4,
    )
    return image


def main(input_file_path, level, output_file_path):
    image = load_image(input_file_path, level)
    image_normalized = normalize(image)
    np.save(output_file_path, image_normalized)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error(
            "Usage: python preprocess_images.py <input_file_path> <level> <fixed_or_moving>"
        )
        sys.exit(1)

    input_file_path = sys.argv[1]
    level = int(sys.argv[2])
    fixed_or_moving = sys.argv[3]

    output_file_name = f"{fixed_or_moving}_normalized_level_{level}.npy"
    output_file_path = Path(input_file_path).parent / output_file_name

    main(
        input_file_path=input_file_path, level=level, output_file_path=output_file_path
    )
