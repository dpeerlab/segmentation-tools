import os
import subprocess
import tempfile
from pathlib import Path
import cv2
from skimage import img_as_uint
import tifffile
import numpy as np
from segmentation_tools.logger import logger


def normalize_image(img: np.ndarray) -> np.ndarray:
    logger.info("Starting normalization: quantile clipping and CLAHE")

    # 1. Clip extreme values (remove top 0.01% brightest pixels)
    low, high = np.quantile(img, [0, 0.9999])
    logger.debug(f"Clipping range: {low:.2f} to {high:.2f}")
    img_clipped = np.clip(img, low, high)

    # 2. Convert to uint16 for OpenCV CLAHE
    img_uint16 = img_as_uint(img_clipped / high)  # normalize to [0, 1] first

    # 3. Apply CLAHE (local histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))
    img_eq = clahe.apply(img_uint16)

    logger.success("Normalization complete")
    return img_eq


def is_tiff_file(file_path: str | Path) -> bool:
    """Check if a file is a valid TIFF."""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    try:
        with tifffile.TiffFile(file_path):
            return True
    except:
        return False
    


def convert_to_tiff(input_path: str | Path, output_root: str | Path) -> Path:
    """Convert a single file or directory of files to TIFF format."""
    input_path = Path(input_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        output_file = output_root / f"{input_path.stem}.tiff"
        _convert_to_tiff_helper(input_path, output_file)
        return output_file
    else:
        raise ValueError(f"Input must be a file or directory: {input_path}")


def _convert_to_tiff_helper(input_file: Path, output_file: Path):
    """Convert a microscopy image to TIFF format using bioformats2raw and raw2ometiff."""
    logger.info(f"Processing: {input_file}")

    with tempfile.TemporaryDirectory() as raw_dir:
        raw_dir_path = Path(raw_dir)

        try:
            subprocess.run(
                [
                    "bioformats2raw",
                    str(input_file),
                    str(raw_dir_path),
                    "--overwrite",
                    "--progress",
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error(f"bioformats2raw failed on {input_file}")
            return

        if not (raw_dir_path / ".zgroup").exists():
            logger.error(f"bioformats2raw output missing .zgroup for {input_file}")
            return

        try:
            subprocess.run(
                ["raw2ometiff", str(raw_dir_path), str(output_file), "--progress"],
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error(f"raw2ometiff failed on {input_file}")
            return

        logger.success(f"Saved TIFF: {output_file}")
    return
