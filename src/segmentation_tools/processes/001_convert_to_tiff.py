import subprocess
import tempfile
import sys
import shutil
from pathlib import Path
from typing import Union
from loguru import logger
import tifffile
from segmentation_tools.utils.config import CHECKPOINT_DIR_NAME
import argparse


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

def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Convert a file to TIFF format, specifying paths and a prefix."
    )

    # Define the arguments as named flags
    parser.add_argument(
        "--input-path",
        required=True,
        type=str,
        help="The path to the input file to be converted.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=str,
        help="The root directory where the output file will be saved.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        type=str,
        help="A prefix to use for the output TIFF file name.",
    )

    return parser.parse_args()


def main(input_file_path, output_root, prefix):
    # 3. Construct the output path
    output_file_name = Path(output_root) / Path(CHECKPOINT_DIR_NAME) / Path(f"{prefix}.tiff")
    
    # 4. Execute the conversion
    convert_file_to_tiff(
        input_file_path=Path(input_file_path),
        output_file_path=output_file_name,
    )
    logger.info("Conversion pipeline complete.")


if __name__ == "__main__":
    # 1. Parse Arguments using named flags
    args = parse_arguments()

    # 2. Access arguments by their names
    input_file_path = args.input_path
    output_root = args.output_root
    prefix = args.prefix

    main(
        input_file_path=input_file_path,
        output_root=output_root,
        prefix=prefix,
    )
    
