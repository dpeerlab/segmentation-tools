import os
import subprocess
import tempfile
from pathlib import Path

import tifffile

from segmentation_tools.logger import logger


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


def convert_to_tiff_if_needed(
    input_file_path: str | Path, intermediates_dir: str | Path = None
):
    """
    Convert the input file to TIFF format if it is not already a TIFF.
    Uses bioformats2raw and raw2ometiff for conversion.
    Updates self.if_file to the converted TIFF path if conversion occurs.
    """
    input_file = Path(input_file_path)

    if is_tiff_file(input_file):
        logger.info(f"Input file is already a TIFF: {input_file}")
        return input_file

    logger.info(f"Converting {input_file} to TIFF format")

    output_root = Path(intermediates_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"{input_file.stem}.tiff"

    with tempfile.TemporaryDirectory() as raw_dir:
        raw_dir_path = Path(raw_dir)

        # Step 1: bioformats2raw
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

        # Step 2: raw2ometiff
        try:
            subprocess.run(
                ["raw2ometiff", str(raw_dir_path), str(output_file), "--progress"],
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error(f"raw2ometiff failed on {input_file}")
            return

    logger.success(f"Converted {input_file.name} to TIFF in {output_root}")
    return output_file
