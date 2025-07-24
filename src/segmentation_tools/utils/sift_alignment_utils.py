from skimage.transform import AffineTransform, ProjectiveTransform
from typing import List
import geopandas as gpd
import pandas as pd
import scanpy as sc
import scipy as sp
import numpy as np
import cv2
from segmentation_tools.logger import logger
import tifffile
from pathlib import Path

from skimage import img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from skimage.transform import warp

from concurrent.futures import ProcessPoolExecutor


def get_best_common_level(
    img1_path: str | Path, img2_path: str | Path, min_size: int = 2048
) -> int:
    """
    Determine the lowest resolution pyramid level (largest index) such that
    both images have at least `min_size` pixels in both width and height.

    Args:
        img1_path: Path to first pyramidal TIFF file.
        img2_path: Path to second pyramidal TIFF file.
        min_size: Minimum width and height required at that level.

    Returns:
        int: Best level index (0 = highest resolution).
    """

    img1_series = tifffile.TiffFile(img1_path).series[0].levels
    img2_series = tifffile.TiffFile(img2_path).series[0].levels

    # Get list of shapes for each level
    levels1 = [page.shape for page in img1_series]
    levels2 = [page.shape for page in img2_series]

    level1 = find_max_valid_level(levels1, min_size)
    level2 = find_max_valid_level(levels2, min_size)

    return level1, level2


def find_max_valid_level(levels: list, min_size: int) -> int:
    """
    Find the lowest resolution level index that still has at least min_size in both dimensions.

    Args:
        levels: List of shapes for each level.
        min_size: Minimum size for width and height.

    Returns:
        int: Index of the valid level (0 = highest resolution).
    """
    for i in reversed(range(len(levels))):
        shape = levels[i]
        if len(shape) == 3:
            _, h, w = shape
        elif len(shape) == 2:
            h, w = shape
        else:
            raise ValueError(f"Unexpected shape: {shape}")

        if h >= min_size and w >= min_size:
            return i
    return 0


def get_SIFT_homography(
    img_fxd,  # Reference image to align to
    img_mvg,  # Image to transform
    min_match_count: int = 10,
    save_img_path=None,
):
    FLANN_INDEX_KDTREE = 1

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # Find the keypoints and descriptors with SIFT
    kp_fxd, dsc_fxd = sift.detectAndCompute(img_fxd, None)
    kp_mvg, dsc_mvg = sift.detectAndCompute(img_mvg, None)
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsc_fxd, dsc_mvg, k=2)
    # Store all the good matches as per Lowe's ratio test.
    good_matches = list(filter(lambda m: m[0].distance < 0.7 * m[1].distance, matches))

    if save_img_path is not None:
        img_matches = cv2.drawMatchesKnn(
            img_fxd,
            kp_fxd,
            img_mvg,
            kp_mvg,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(save_img_path, img_matches)

    # Compute 3x3 homography matrix
    if len(good_matches) >= min_match_count:
        fxd_pts = np.array([kp_fxd[m.queryIdx].pt for m, _ in good_matches])
        mvg_pts = np.array([kp_mvg[m.trainIdx].pt for m, _ in good_matches])
        M, mask = cv2.findHomography(mvg_pts, fxd_pts, cv2.RANSAC, 5.0)
        return ProjectiveTransform(M)
    else:
        msg = (
            f"Insufficient no. matches detected to perform image "
            f"registration: {len(good_matches)} matches"
        )
        raise AssertionError(msg)


def _generate_img_variants(image: np.ndarray):
    """Return list of (transformed image, name, transform) tuples."""
    variants = [
        (image, "original", np.eye(3)),
        (np.rot90(image, 1), "rot90", AffineTransform(rotation=np.pi / 2).params),
        (np.rot90(image, 2), "rot180", AffineTransform(rotation=np.pi).params),
        (np.rot90(image, 3), "rot270", AffineTransform(rotation=3 * np.pi / 2).params),
        (np.fliplr(image), "flip_lr", AffineTransform(scale=(-1, 1)).params),
        (np.flipud(image), "flip_ud", AffineTransform(scale=(1, -1)).params),
        (
            np.rot90(np.fliplr(image), 1),
            "flip_lr_rot90",
            (
                AffineTransform(rotation=np.pi / 2) + AffineTransform(scale=(-1, 1))
            ).params,
        ),
        (
            np.rot90(np.flipud(image), 1),
            "flip_ud_rot90",
            (
                AffineTransform(rotation=np.pi / 2) + AffineTransform(scale=(1, -1))
            ).params,
        ),
    ]
    return variants


def _score_variant(variant_img, name, pre_matrix, fxd_img, save_img_path):
    try:
        fxd_img = img_as_ubyte(fxd_img)
        variant_img = img_as_ubyte(variant_img)

        save_path = f"{save_img_path}_{name}.png" if save_img_path else None
        H = get_SIFT_homography(
            img_fxd=fxd_img, img_mvg=variant_img, save_img_path=save_path
        )

        aligned = warp(
            variant_img,
            inverse_map=H.inverse,
            output_shape=fxd_img.shape,
            preserve_range=True,
        )

        score = ssim(fxd_img, aligned, data_range=aligned.max() - aligned.min())
        combined_matrix = H.params @ pre_matrix
        logger.info(f"Variant: {name}, Score: {score:.4f}")
        return (score, name, combined_matrix)

    except Exception as e:
        return (-1, name, None)  # Failed variant


def find_best_sift(mvg_img, fxd_img, save_img_path=None):
    mvg_img = img_as_ubyte(mvg_img)
    fxd_img = img_as_ubyte(fxd_img)

    variants = _generate_img_variants(mvg_img)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _score_variant, variant_img, name, pre_matrix, fxd_img, save_img_path
            )
            for variant_img, name, pre_matrix in variants
        ]

        results = [f.result() for f in futures]

    # Pick the best
    best = max(results, key=lambda r: r[0])  # r[0] = score
    best_score, best_name, best_matrix = best

    if best_matrix is None:
        raise RuntimeError("All variant alignments failed.")

    logger.info(f"Best alignment found: {best_name} with score {best_score:.4f}")
    return ProjectiveTransform(matrix=best_matrix)
