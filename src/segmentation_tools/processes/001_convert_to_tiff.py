import subprocess
import tempfile
import sys
import shutil
from pathlib import Path
from typing import Union
from loguru import logger
import tifffile
from segmentation_tools.utils.config import CHECKPOINT_DIR_NAME
from icecream import ic


# --- Internal Helper Functions ---


def _is_tiff_file(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid TIFF."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    try:
        with tifffile.TiffFile(file_path):
            return True
    except Exception:
        return False


def convert_file_to_tiff(input_file_path: Path, output_file_path: Path):
    """Convert a single microscopy image to TIFF using bioformats2raw and raw2ometiff."""
    if _is_tiff_file(input_file_path):
        logger.info(f"Skipping conversion: {input_file_path.name} is already a TIFF.")
        logger.info(f"Copying {input_file_path} to {output_file_path}")
        shutil.copy(input_file_path, output_file_path)
        return

    logger.info(f"Converting: {input_file_path.name}")

    with tempfile.TemporaryDirectory() as raw_dir:
        raw_dir_path = Path(raw_dir)

        # Step 1: bioformats2raw
        try:
            logger.info("Running bioformats2raw...")
            subprocess.run(
                [
                    "bioformats2raw",
                    str(input_file_path),
                    str(raw_dir_path),
                    "--overwrite",
                    "--progress",
                ],
                check=True,
                capture_output=True,  # Capture output to prevent flooding console
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"bioformats2raw failed on {input_file_path.name}. Error: {e.stderr}"
            )
            return False

        if not (raw_dir_path / ".zgroup").exists():
            logger.error(
                f"bioformats2raw output missing .zgroup for {input_file_path.name}"
            )
            return False

        # Step 2: raw2ometiff
        try:
            logger.info("Running raw2ometiff...")
            subprocess.run(
                ["raw2ometiff", str(raw_dir_path), str(output_file_path), "--progress"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"raw2ometiff failed on {input_file_path.name}. Error: {e.stderr}"
            )
            return False

    logger.success(f"Successfully converted and saved TIFF: {output_file_path.name}")
    return True


# --- Command-Line Entry Point ---

# Parameters
# input_path (file or dir to convert)
# output_root (dir to save file to)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(len(sys.argv))
        print(sys.argv)
        print("Usage: python tiff_converter.py <input_path> <output_root> <prefix>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_root = sys.argv[2]
    prefix = sys.argv[3]

    output_file_name = Path(output_root) / Path(CHECKPOINT_DIR_NAME) / Path(f"{prefix}.tiff")
    try:
        convert_file_to_tiff(
            input_file_path=Path(input_file_path),
            output_file_path=Path(output_file_name),
        )
        logger.info("Conversion pipeline complete.")
    except Exception as e:
        logger.error(
            "Usage: python tiff_converter.py <input_file_path> <output_root> <prefix>"
        )
        logger.error(f"An unexpected error occurred in the pipeline: {e}")
        sys.exit(1)
