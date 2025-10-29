from icecream import ic
from pathlib import Path
from loguru import logger
from segmentation_tools.utils.config import CHECKPOINT_DIR_NAME, RESULTS_DIR_NAME
import os
import shutil
import sys


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
    if len(sys.argv) != 3:
        logger.error("Usage: python setup_directories.py <output_dir> <job_title>")
        sys.exit(1)

    output_dir_root = sys.argv[1]
    job_title = sys.argv[2]

    output_dir, processed_tiff_dir, checkpoint_dir = main(
        output_dir_root=Path(output_dir_root), job_title=job_title
    )
