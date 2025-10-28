import tifffile
import cv2
from pathlib import Path
from typing import Union
from loguru import logger

import sys


def find_optimal_sift_level_by_keypoints(
    moving_file: Union[str, Path],
    fixed_file: Union[str, Path],
    k_min: int,
    k_max: int,
    max_level_search: int,
    min_level_search: int,
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
        len(tif_m.series[0].levels) - 1,
        len(tif_f.series[0].levels) - 1,
        max_level_search,
    )

    sift = cv2.SIFT_create()
    best_level = 0

    # Iterate from a coarse level (highest index) down to Level 0
    for i in range(max_possible_level, -1, -1):

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

        # --- Keypoint Check ---

        # We only need the keypoint count, not the descriptors
        kp_m = sift.detect(img_m, None)
        kp_f = sift.detect(img_f, None)

        count_m = len(kp_m)
        count_f = len(kp_f)

        logger.info(f"Level {i}: Keypoints (M={count_m}, F={count_f})")

        # 1. Check if both images have too many keypoints (coarse-to-fine step)
        if count_m > k_max and count_f > k_max:
            # If too many, keep iterating (go coarser/higher index)
            continue

        # 2. Check if both images have enough keypoints
        if count_m >= k_min and count_f >= k_min:
            # This is the coarsest level that meets the k_min requirement for both.
            best_level = i
            return best_level

    best_level = max(min_level_search, best_level)

    # If the loop completes without finding a suitable coarse level,
    # it means only Level 0 or a very low level can meet k_min.
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


if __name__ == "__main__":
    moving_fp = sys.argv[1]
    fixed_fp = sys.argv[2]
    k_min = int(sys.argv[3])
    k_max = int(sys.argv[4])
    max_level_search = int(sys.argv[5])
    min_level_search = int(sys.argv[6])

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
