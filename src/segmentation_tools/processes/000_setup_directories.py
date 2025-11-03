from icecream import ic
from pathlib import Path
from loguru import logger
from segmentation_tools.utils.config import CHECKPOINT_DIR_NAME, RESULTS_DIR_NAME
import os
import shutil
import sys
import argparse


def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Set up output directories for segmentation tools."
    )

    # Define the arguments as named flags
    parser.add_argument(
        "--output-root",
        required=True,
        type=str,
        help="The root directory where the output file will be saved.",
    )
    parser.add_argument(
        "--job-title",
        required=True,
        type=str,
        help="The title of the job being processed.",
    )

    return parser.parse_args()


def main(output_dir_root: Path, job_title: str):
    """Set up output directories."""
    output_dir_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir_root / job_title

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = output_dir / RESULTS_DIR_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, results_dir, checkpoint_dir


if __name__ == "__main__":
    args = parse_arguments()
    output_dir_root = args.output_root
    job_title = args.job_title

    output_dir, processed_tiff_dir, checkpoint_dir = main(
        output_dir_root=Path(output_dir_root), job_title=job_title
    )
