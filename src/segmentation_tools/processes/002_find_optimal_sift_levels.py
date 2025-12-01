import tifffile
import cv2
from pathlib import Path
from typing import Union
from loguru import logger
import argparse

import sys

def find_optimal_sift_level_by_keypoints(
    moving_file: Union[str, Path],
    fixed_file: Union[str, Path],
    k_min: int,
    k_max: int,
    max_level_search: int = None,
    min_level_search: int = None,
) -> int:
    """
    Finds the coarsest common pyramid level that yields a sufficient number
    of SIFT keypoints (k_min <= K <= k_max) in both images.
    """
    try:
        tif_m = tifffile.TiffFile(moving_file)
        tif_f = tifffile.TiffFile(fixed_file)
    except Exception as e:
        logger.error(f"Error opening TIFF files: {e}")
        return 0

    max_possible_level = min(
        len(tif_m.series[0].levels) - 1, len(tif_f.series[0].levels) - 1
    )

    if max_level_search is not None:
        max_possible_level = min(max_possible_level, max_level_search)
    if min_level_search is not None:
        min_level_search = max(min_level_search, 0)
    else:
        min_level_search = 0

    sift = cv2.SIFT_create()
    best_level = 0

    # Iterate from a coarse level (highest index) down to Level 0
    for i in range(max_possible_level, min_level_search - 1, -1):
        print(i)
        # Load the image data for this level
        # Note: tifffile.TiffPage.asarray() loads the image into memory (NumPy)
        img_m_page = tif_m.series[0].levels[i]
        img_f_page = tif_f.series[0].levels[i]

        # Read the image content as uint8 for SIFT
        # Handle potential 3D (C, H, W) or 4D (Z, C, H, W) to get a 2D slice
        img_m = img_m_page.asarray().squeeze()
        img_f = img_f_page.asarray().squeeze()

        # SIFT requires 8-bit image data
        if img_m.ndim > 2:
            img_m = img_m[0]  # Take first channel/slice if multichannel
        if img_f.ndim > 2:
            img_f = img_f[0]

        img_m = cv2.normalize(img_m, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_f = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


        # We only need the keypoint count, not the descriptors
        kp_m = sift.detect(img_m, None)
        kp_f = sift.detect(img_f, None)

        count_m = len(kp_m)
        count_f = len(kp_f)

        logger.info(f"Level {i}: Keypoints (M={count_m}, F={count_f})")

        best_level = i
        if count_m < k_min or count_f < k_min:
            continue

        if count_m > k_max or count_f > k_max:
            return best_level + 1

        if k_min < count_m < k_max and k_min < count_f < k_max:
            return best_level
        
    return best_level


def main(moving_fp, fixed_fp, k_min, k_max, max_level_search, min_level_search, checkpoint_dir):
    best_level = find_optimal_sift_level_by_keypoints(
        moving_fp, fixed_fp, k_min, k_max, max_level_search, min_level_search
    )
    logger.info(f"Optimal SIFT Level: {best_level}")

    output_file_path = checkpoint_dir / "optimal_sift_level.txt"
    with open(output_file_path, "w") as f:
        f.write(str(best_level) + "\n")
    logger.info(f"Optimal level saved to {output_file_path}")
    return best_level

def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Find optimal SIFT pyramid levels for image registration."
    )

    # Define the arguments as named flags
    parser.add_argument(
        "--moving-file",
        required=True,
        type=str,
        help="Path to the moving TIFF file.",
    )
    parser.add_argument(
        "--fixed-file",
        required=True,
        type=str,
        help="Path to the fixed TIFF file.",
    )
    parser.add_argument(
        "--k-min",
        required=True,
        type=int,
        help="Minimum number of SIFT keypoints required.",
    )
    parser.add_argument(
        "--k-max",
        required=True,
        type=int,
        help="Maximum number of SIFT keypoints allowed.",
    )
    parser.add_argument(
        "--max-level-search",
        required=False,
        type=int,
        help="Maximum pyramid level to search.",
    )
    parser.add_argument(
        "--min-level-search",
        required=False,
        type=int,
        help="Minimum pyramid level to search.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    moving_fp = args.moving_file
    fixed_fp = args.fixed_file
    k_min = args.k_min
    k_max = args.k_max
    max_level_search = args.max_level_search
    min_level_search = args.min_level_search

    checkpoint_dir = Path(moving_fp).parent

    main(
        moving_fp=moving_fp,
        fixed_fp=fixed_fp,
        k_min=k_min,
        k_max=k_max,
        max_level_search=max_level_search,
        min_level_search=min_level_search,
        checkpoint_dir=checkpoint_dir,
    )
