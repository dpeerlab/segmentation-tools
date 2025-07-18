import os
import subprocess
import tempfile
from pathlib import Path

import skimage
import tifffile
import numpy as np
from segmentation_tools.logger import logger
from numpy.typing import ArrayLike
from skimage.transform import (
    ProjectiveTransform, AffineTransform, EuclideanTransform
)

def get_level_transform(
    tiff_object: os.PathLike,
    level_to: int,
    level_from: int = 0,
):
    lvl = tiff_object.series[0].levels[level_from]
    from_dims = dict(zip(lvl.axes, lvl.shape))
    lvl = tiff_object.series[0].levels[level_to]
    to_dims = dict(zip(lvl.axes, lvl.shape))
    scale = to_dims['X'] / from_dims['X'], to_dims['Y'] / from_dims['Y']
    return AffineTransform(scale=scale)


def normalize(
    img: np.ndarray,
    quantiles: ArrayLike = [0.01, 0.99],
    values: ArrayLike = None,
):
    # Clip values
    if values is None:
        # Clip to quantile
        qtl = np.quantile(img, quantiles, axis=(-2, -1))
        qtl = qtl.reshape((*qtl.shape, 1, 1))
        img = np.clip(img, a_min=qtl[0], a_max=qtl[1])

    else:
        # Clip to provided values
        img = np.clip(img, a_min=values[0], a_max=values[1])
    
    # Rescale
    mins = img.min(axis=(-2, -1))
    mins = mins.reshape((*mins.shape, 1, 1))
    maxs = img.max(axis=(-2, -1))
    maxs = maxs.reshape((*maxs.shape, 1, 1))
    img = (img - mins) / (maxs - mins)
    # Convert to [0, 255]
    return skimage.util.img_as_ubyte(img)

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
