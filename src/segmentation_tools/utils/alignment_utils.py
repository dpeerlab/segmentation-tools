from skimage.transform import AffineTransform, ProjectiveTransform
from typing import List
import geopandas as gpd
import pandas as pd
import scanpy as sc
import scipy as sp
import numpy as np
import cv2
from segmentation_tools.logger import logger


from skimage import img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from skimage.transform import warp

def _generate_img_variants(image: np.ndarray):
    """Return list of (transformed image, name) tuples."""
    return [
        (image, "original"),
        (np.rot90(image, 1), "rot90"),
        (np.rot90(image, 2), "rot180"),
        (np.rot90(image, 3), "rot270"),
        (np.fliplr(image), "flip_lr"),
        (np.flipud(image), "flip_ud"),
        (np.rot90(np.fliplr(image), 1), "flip_lr_rot90"),
        (np.rot90(np.flipud(image), 1), "flip_ud_rot90"),
    ]

def find_best_alignment(mvg_img, fxd_img):
    best_score = -1
    best_result = None
    best_transform_name = None

    mvg_img = img_as_ubyte(mvg_img)
    fxd_img = img_as_ubyte(fxd_img)


    # H = get_SIFT_homography(img_fxd=fxd_img, img_mvg=mvg_img)
    for variant, name in _generate_img_variants(mvg_img):
        try:
            H = get_SIFT_homography(img_fxd=fxd_img, img_mvg=variant, save_img_path=f"/data1/peerd/ghoshr/segmentation_tools/data/outputs/intermediates/sift_matches_{name}.png")
            aligned = warp(
                variant,
                inverse_map=H.inverse,
                output_shape=fxd_img.shape,
                preserve_range=True,
            )

            score = ssim(fxd_img, aligned, data_range=aligned.max() - aligned.min())
            logger.info(f"Alignment score for '{name}': {score:.4f}")
            if score > best_score:
                best_score = score
                best_result = aligned
                best_transform_name = name
        except Exception:
            logger.warning(f"Failed to align image variant '{name}'")
            continue

    return best_result, best_transform_name, best_score


def get_SIFT_homography(
    img_fxd, # Reference image to align to
    img_mvg, # Image to transform
    min_match_count: int = 10,
    save_img_path = None,
):
    FLANN_INDEX_KDTREE = 1

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # Find the keypoints and descriptors with SIFT
    kp_fxd, dsc_fxd = sift.detectAndCompute(img_fxd, None)
    kp_mvg, dsc_mvg = sift.detectAndCompute(img_mvg, None)
    index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
    search_params = {'checks': 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsc_fxd, dsc_mvg, k=2)
    # Store all the good matches as per Lowe's ratio test.
    good_matches = list(filter(
        lambda m: m[0].distance < 0.7 * m[1].distance, 
        matches
    ))

    if save_img_path is not None:
        img_matches = cv2.drawMatchesKnn(
            img_fxd, kp_fxd,
            img_mvg, kp_mvg,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
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
            f'Insufficient no. matches detected to perform image '
            f'registration: {len(good_matches)} matches'
        )
        raise AssertionError(msg)
    