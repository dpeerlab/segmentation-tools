import os
import subprocess
import tempfile
from pathlib import Path

import tifffile

from segmentation_tools.logger import logger


def _convert_to_tiff(input_file: Path, output_file: Path):
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


def _is_tiff_file(file_path: str) -> bool:
    """Check if a file is a valid TIFF."""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    try:
        with tifffile.TiffFile(file_path):
            logger.info(f"Valid TIFF file: {file_path}")
            return True
    except Exception:
        logger.warning(f"Not a valid TIFF file: {file_path}")
        return False


def convert_to_tiff(input_path: str, output_root: str):
    """Convert a single file or directory of files to TIFF format."""
    input_path = Path(input_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        output_file = output_root / f"{input_path.stem}.tiff"
        _convert_to_tiff(input_path, output_file)

    elif input_path.is_dir():
        for file in input_path.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(input_path)
                output_subdir = output_root / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_file = output_subdir / f"{file.stem}.tiff"
                _convert_to_tiff(file, output_file)

    else:
        raise ValueError(f"Input must be a file or directory: {input_path}")
