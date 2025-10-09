import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import cupy as cp
import cucim

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte

from skimage.metrics import structural_similarity as ssim
from skimage.transform import AffineTransform, ProjectiveTransform

import mirage
from segmentation_tools.logger import logger


import tifffile 
from icecream import ic


# def get_SIFT_homography(
#     image_fixed,  # Reference image to align to
#     image_moving,  # Image to transform
#     min_match_count: int = 10,
# ):
#     FLANN_INDEX_KDTREE = 1

#     # Initiate SIFT detector
#     sift = cv2.SIFT_create()
#     # Find the keypoints and descriptors with SIFT
#     kp_fixed, dsc_fixed = sift.detectAndCompute(image_fixed, None)
#     kp_moving, dsc_moving = sift.detectAndCompute(image_moving, None)
#     index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
#     search_params = {"checks": 50}
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(dsc_fixed, dsc_moving, k=2)
#     # Store all the good matches as per Lowe's ratio test.
#     good_matches = list(filter(lambda m: m[0].distance < 0.7 * m[1].distance, matches))

#     # Compute 3x3 homography matrix
#     if len(good_matches) >= min_match_count:
#         fixed_pts = np.array([kp_fixed[m.queryIdx].pt for m, _ in good_matches])
#         moving_pts = np.array([kp_moving[m.trainIdx].pt for m, _ in good_matches])
#         M, _ = cv2.findHomography(moving_pts, fixed_pts, cv2.RANSAC, 5.0)
#         return ProjectiveTransform(M)
#     else:
#         msg = (
#             f"Insufficient no. matches detected to perform image "
#             f"registration: {len(good_matches)} matches"
#         )
#         raise AssertionError(msg)


# def _generate_image_variants(image: np.ndarray):
#     """Return list of (transformed image, name, transform) tuples."""
#     variants = [
#         (image, "original", np.eye(3)),
#         (np.rot90(image, 1), "rot90", AffineTransform(rotation=np.pi / 2).params),
#         (np.rot90(image, 2), "rot180", AffineTransform(rotation=np.pi).params),
#         (np.rot90(image, 3), "rot270", AffineTransform(rotation=3 * np.pi / 2).params),
#         (np.fliplr(image), "flip_lr", AffineTransform(scale=(-1, 1)).params),
#         (np.flipud(image), "flip_ud", AffineTransform(scale=(1, -1)).params),
#         (
#             np.rot90(np.fliplr(image), 1),
#             "flip_lr_rot90",
#             (
#                 AffineTransform(rotation=np.pi / 2) + AffineTransform(scale=(-1, 1))
#             ).params,
#         ),
#         (
#             np.rot90(np.flipud(image), 1),
#             "flip_ud_rot90",
#             (
#                 AffineTransform(rotation=np.pi / 2) + AffineTransform(scale=(1, -1))
#             ).params,
#         ),
#     ]
#     return variants


# def _score_variant(variant_image, name, pre_matrix, fixed_image):
#     try:
#         fixed_image = img_as_ubyte(fixed_image)
#         variant_image = img_as_ubyte(variant_image)

#         ic("getting H")
#         H = get_SIFT_homography(
#             image_fixed=fixed_image,
#             image_moving=variant_image,
#         )
#         ic("warping")
#         aligned = cucim.skimage.transform.warp(
#             cp.asarray(variant_image),
#             inverse_map=cp.asarray(H.inverse.params),
#             output_shape=cp.asarray(fixed_image.shape[:2]),
#             preserve_range=True,
#             order = 5,
#         )

#         ic("get ssim")
#         score = cucim.skimage.metrics.structural_similarity(cp.asarray(fixed_image), aligned, data_range=aligned.max() - aligned.min())
#         combined_matrix = H.params @ pre_matrix
#         logger.info(f"Variant: {name}, Score: {score:.4f}")
#         return (score, name, combined_matrix)

#     except Exception as e:
#         logger.warning(f"Failed to score variant {name}: {e}")
#         return (-1, name, None)  # Failed variant


# def find_best_sift(moving_image, fixed_image):
#     moving_image = img_as_ubyte(moving_image)
#     fixed_image = img_as_ubyte(fixed_image)

#     variants = _generate_image_variants(moving_image)

#     with ProcessPoolExecutor() as executor:
#         futures = [
#             executor.submit(
#                 _score_variant,
#                 variant_image,
#                 name,
#                 pre_matrix,
#                 fixed_image,
#             )
#             for variant_image, name, pre_matrix in variants
#         ]

#         results = [f.result() for f in futures]
#     # Pick the best
#     best = max(results, key=lambda r: r[0])  # r[0] = score
#     best_score, best_name, best_matrix = best

#     if best_matrix is None:
#         raise RuntimeError("All variant alignments failed.")

#     logger.info(f"Best alignment found: {best_name} with score {best_score:.4f}")
#     return ProjectiveTransform(matrix=np.linalg.inv(best_matrix))

# Helper to perform SIFT matching and return key components
# NOTE: This replaces the core of the original get_SIFT_homography
def _sift_match_core(image_fixed, image_moving, min_match_count: int = 10):
    FLANN_INDEX_KDTREE = 1

    ic("sift detect")
    sift = cv2.SIFT_create()
    kp_fixed, dsc_fixed = sift.detectAndCompute(image_fixed, None)
    kp_moving, dsc_moving = sift.detectAndCompute(image_moving, None)

    if dsc_fixed is None or dsc_moving is None:
        return 0, None, None, None, None # Return 0 matches if descriptors are missing

    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    ic("flann match")
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsc_fixed, dsc_moving, k=2)

    ic("filter matches")
    good_matches = list(filter(lambda m: m[0].distance < 0.7 * m[1].distance, matches))
    match_count = len(good_matches)

    ic("matches found", match_count)
    if match_count >= min_match_count:
        fixed_pts = np.array([kp_fixed[m.queryIdx].pt for m, _ in good_matches])
        moving_pts = np.array([kp_moving[m.trainIdx].pt for m, _ in good_matches])
        return match_count, fixed_pts, moving_pts, kp_fixed, kp_moving
    else:
        return 0, None, None, None, None

# --- Minimal Change to get_SIFT_homography (Uses the new core) ---
def get_SIFT_homography(
    image_fixed,  # Reference image to align to
    image_moving,  # Image to transform
    min_match_count: int = 10,
):
    match_count, fixed_pts, moving_pts, _, _ = _sift_match_core(
        image_fixed, image_moving, min_match_count
    )

    # Compute 3x3 homography matrix
    if match_count >= min_match_count:
        M, _ = cv2.findHomography(moving_pts, fixed_pts, cv2.RANSAC, 5.0)
        return ProjectiveTransform(M)
    else:
        msg = (
            f"Insufficient no. matches detected to perform image "
            f"registration: {match_count} matches"
        )
        raise AssertionError(msg)
# ------------------------------------------------------------------

# --- New Function for Pass 1: Match Counting (No Homography/SSIM) ---
def _match_count_variant(variant_image, name, pre_matrix, fixed_image):
    try:
        fixed_image = img_as_ubyte(fixed_image)
        variant_image = img_as_ubyte(variant_image)

        # Only run the matching core, skip findHomography
        match_count, _, _, _, _ = _sift_match_core(
            image_fixed=fixed_image,
            image_moving=variant_image,
        )
        logger.info(f"Variant Match Count: {name}, Matches: {match_count}")
        # Return match_count as the score for Pass 1
        return (match_count, name, pre_matrix)

    except Exception as e:
        logger.warning(f"Failed to count matches for variant {name}: {e}")
        return (-1, name, None)
# ------------------------------------------------------------------

# --- No change to _generate_image_variants ---
def _generate_image_variants(image: np.ndarray):
    """Return list of (transformed image, name, transform) tuples."""
    # ... (Original code here)
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
# ------------------------------------------------------------------

# --- No change to _score_variant (It's only called on the best variant now) ---
def _score_variant(variant_image, name, pre_matrix, fixed_image):
    try:
        fixed_image = img_as_ubyte(fixed_image)
        variant_image = img_as_ubyte(variant_image)

        ic("getting H")
        H = get_SIFT_homography(
            image_fixed=fixed_image,
            image_moving=variant_image,
        )
        ic("warping")
        aligned = cucim.skimage.transform.warp(
            cp.asarray(variant_image),
            inverse_map=cp.asarray(H.inverse.params),
            output_shape=cp.asarray(fixed_image.shape[:2]),
            preserve_range=True,
            order = 5,
        )

        ic("get ssim")
        score = cucim.skimage.metrics.structural_similarity(cp.asarray(fixed_image), aligned, data_range=aligned.max() - aligned.min())
        combined_matrix = H.params @ pre_matrix
        logger.info(f"Variant: {name}, Score: {score:.4f}")
        return (score, name, combined_matrix)

    except Exception as e:
        logger.warning(f"Failed to score variant {name}: {e}")
        print(e)
        return (-1, name, None)  # Failed variant
# ------------------------------------------------------------------

# --- Refactored find_best_sift (Two-Pass Logic) ---
def find_best_sift(moving_image, fixed_image):
    moving_image = img_as_ubyte(moving_image)
    fixed_image = img_as_ubyte(fixed_image)

    variants = _generate_image_variants(moving_image)

    # --- PASS 1: Find best orientation (max SIFT matches) ---
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _match_count_variant,  # Use the fast match counting function
                variant_image,
                name,
                pre_matrix,
                fixed_image,
            )
            for variant_image, name, pre_matrix in variants
        ]
        pre_results = [f.result() for f in futures]
    
    ic()
    # Pick the best pre-alignment based on match count (r[0])
    best_pre = max(pre_results, key=lambda r: r[0])
    best_match_count, best_name, best_pre_matrix = best_pre
    
    if best_match_count <= 0:
        raise RuntimeError("No variant found with sufficient SIFT matches.")

    logger.info(f"Best pre-alignment found: {best_name} with {best_match_count} matches.")

    # --- PASS 2: Compute full SIFT Homography and SSIM only for the best variant ---
    # Find the image data for the best variant to pass to the final scorer
    best_variant_image = next(
        v_img for v_img, v_name, _ in variants if v_name == best_name
    )

    # Compute the final, accurate score, homography, and matrix
    best_score, final_best_name, best_matrix = _score_variant(
        best_variant_image,
        best_name,
        best_pre_matrix,
        fixed_image,
    )
    
    # Note: _score_variant already logs the score/name internally
    if best_matrix is None or best_score == -1:
        raise RuntimeError(f"Final alignment failed for best variant: {best_name}.")

    logger.info(f"Best alignment found: {final_best_name} with score {best_score:.4f}")
    return ProjectiveTransform(matrix=np.linalg.inv(best_matrix))
# ------------------------------------------------------------------


def convert_transform_to_dense_map(
    transform: AffineTransform, output_shape: tuple
) -> np.ndarray:
    """
    Converts an AffineTransform object into a dense, per-pixel inverse
    coordinate map array compatible with skimage.transform.warp.

    Args:
        transform (AffineTransform): The AffineTransform object.
        output_shape (tuple): The desired shape (H, W) of the output image.

    Returns:
        np.ndarray: A dense inverse coordinate map of shape (H, W, 2).
                    dense_map[r, c] contains the source (r_src, c_src)
                    for the destination pixel at (r, c).
    """

    # 1. Determine the output grid size
    H, W = output_shape

    # 2. Create the coordinate grid for the destination (output) image
    # The grid represents every pixel (r, c) in the output
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Stack them into a 2D array of (N, 2) coordinates, where N = H * W
    # The shape is (H*W, 2)
    destination_coords = np.stack([rr.ravel(), cc.ravel()], axis=1)

    # 3. Apply the *inverse* transform
    # We use the inverse because the dense map must tell us where to look (source)
    # for a given location (destination).
    source_coords_linear = transform.inverse(destination_coords)  # Shape: (H*W, 2)

    # 4. Reshape the result into a dense map
    # The final shape is (H, W, 2)
    dense_map_from_matrix = source_coords_linear.reshape(H, W, 2)

    return dense_map_from_matrix


def run_mirage(
    moving_image,
    fixed_image,
    bin_mask=None,
    pad=12,
    offset=12,
    num_neurons=300,
    num_layers=3,
    pool=1,
    loss="SSIM",
):
    """
    Run MIRAGE alignment on the given images.

    Parameters:
        moving_image: The moving image to align.
        fixed_image: The fixed image to align against.
        bin_mask: Optional binary mask for the moving image.
        pad: Padding for the model.
        offset: Offset for the model.
        num_neurons: Number of neurons in the model.
        num_layers: Number of layers in the model.
        pool: Pooling factor for the model.
        loss: Loss function to use ("SSIM" or "MSE").
        save_img_dir: Directory to save aligned images (optional).

    Returns:
        Aligned image as a numpy array.
    """
    logger.info("Running MIRAGE alignment...")

    mirage_model = mirage.MIRAGE(
        images=moving_image,
        references=fixed_image,
        bin_mask=bin_mask,
        pad=pad,
        offset=offset,
        num_neurons=num_neurons,
        num_layers=num_layers,
        pool=pool,
        loss=loss,
    )

    logger.info("Training MIRAGE model...")
    try:
        mirage_model.train(batch_size=256, num_steps=512, lr__sched=True, LR=0.005)

        logger.info("MIRAGE model training complete. Computing transform...")

        mirage_model.compute_transform()
    except Exception as e:
        logger.error(f"Error computing transform: {e}")
        return None

    return mirage_model.get_transform()


def get_level_transform(
    path_to_tiff: os.PathLike, level_to: int, level_from: int = 0
) -> AffineTransform:
    """
    Compute the scaling transform (AffineTransform) between two levels of a TIFF pyramid.

    Args:
        path_to_tiff (os.PathLike): Path to the TIFF file.
        level_to (int): Target resolution level.
        level_from (int): Source resolution level (default is 0, i.e. highest res).

    Returns:
        AffineTransform: An affine transform representing the scale between the levels.
    """
    tf = tifffile.TiffFile(path_to_tiff)
    lvl = tf.series[0].levels[level_from]
    from_dims = dict(zip(lvl.axes, lvl.shape))
    lvl = tf.series[0].levels[level_to]
    to_dims = dict(zip(lvl.axes, lvl.shape))
    scale_x = from_dims["X"] / to_dims["X"]
    scale_y = from_dims["Y"] / to_dims["Y"]
    return AffineTransform(scale=(scale_x, scale_y))

def combine_dense_maps(mirage_dense_map: np.ndarray, linear_dense_map: np.ndarray) -> np.ndarray:
    """
    Chains two inverse dense maps (composition).
    
    Args:
        mirage_dense_map (np.ndarray): The first inverse map (H, W, 2). 
                                      mirage_map[r, c] gives an intermediate (r', c').
        linear_dense_map (np.ndarray): The second inverse map (H, W, 2). 
                                       Matrix_map[r', c'] gives the final (r_src, c_src).

    Returns:
        np.ndarray: The composed inverse map (H, W, 2).
    """
    
    H, W, _ = mirage_dense_map.shape
    
    # 1. Get the intermediate source coordinates from the first map (Model)
    intermediate_coords = mirage_dense_map.reshape(-1, 2) 

    # 2. Sample the second map (Matrix) using the intermediate coordinates as indices
    
    # Safely convert the floating-point coordinates from the first map to integers 
    # for indexing the second map. Clamping prevents out-of-bounds indexing.
    H_matrix, W_matrix = linear_dense_map.shape[:2]
    
    r_int_matrix = np.clip(np.round(intermediate_coords[:, 0]).astype(int), 0, H_matrix - 1)
    c_int_matrix = np.clip(np.round(intermediate_coords[:, 1]).astype(int), 0, W_matrix - 1)

    # Look up the final source coordinates: Final Map(r, c) = Matrix Map(Model Map(r, c))
    final_combined_coords = linear_dense_map[r_int_matrix, c_int_matrix]

    # 3. Reshape and return the final dense map
    return final_combined_coords.reshape(H, W, 2)


def main(moving_file_path, fixed_file_path, moving_level, fixed_level, original_moving_file_path, original_fixed_file_path, high_res_level, use_mirage):
    ic(moving_level, fixed_level, high_res_level)
    moving_level = int(moving_level)
    fixed_level = int(fixed_level)
    high_res_level = int(high_res_level)

    ic("loading")
    moving_image = np.load(moving_file_path)
    fixed_image = np.load(fixed_file_path)
    ic("moving to gpu")
    moving_image_cp = cp.array(moving_image).astype("float32")
    fixed_image_cp = cp.array(fixed_image).astype("float32")

    ic("hist match")
    moving_image = cucim.skimage.exposure.match_histograms(
        moving_image_cp, fixed_image_cp
    ).get()

    ic("sift")
    sift_transform = find_best_sift(moving_image, fixed_image)

    ic("linear transform")
    linear_transform = (
        get_level_transform(
            original_moving_file_path,
            level_to=moving_level,
            level_from=high_res_level,
        ).params
        @ sift_transform.params
        @ get_level_transform(
            original_fixed_file_path,
            level_to=high_res_level,
            level_from=fixed_level,
        ).params
    )

    ic("dense map from matrix")
    linear_dense_map = convert_transform_to_dense_map(
        transform=AffineTransform(matrix=linear_transform), output_shape=fixed_image.shape
    )

    ic("mirage")
    if use_mirage:
        mirage_dense_map =run_mirage(
            moving_image,
            fixed_image,
            bin_mask=None,
            pad=12,
            offset=12,
            num_neurons=300,
            num_layers=3,
            pool=1,
            loss="SSIM",
        )

        transform_dense_map = combine_dense_maps(mirage_dense_map=mirage_dense_map, linear_dense_map=linear_dense_map)
    else:
        transform_dense_map = linear_dense_map

    transform_file_name = "transform.npy"
    transform_file_path = Path(moving_file_path).parent / transform_file_name
    np.save(transform_file_path, transform_dense_map)
    logger.info(f"Transformation dense map saved to {transform_file_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 9:
        logger.error(
            "Usage: python 003_find_alignment_transform.py <moving_file> <fixed_file> <moving_level> <fixed_level> <original_moving_file> <original_fixed_file> <high_res_level> <run_mirage>"
        )
        sys.exit(1)

    moving_file_path = sys.argv[1]
    fixed_file_path = sys.argv[2]
    moving_level = sys.argv[3]
    fixed_level = sys.argv[4]
    high_res_level = sys.argv[5]
    original_moving_file_path = sys.argv[6]
    original_fixed_file_path = sys.argv[7]
    use_mirage = sys.argv[8]

    use_mirage = use_mirage.lower() in ("yes", "true", "t", "1")

    main(
        moving_file_path=moving_file_path,
        fixed_file_path=fixed_file_path,

        moving_level=moving_level,
        fixed_level=fixed_level,
        high_res_level=high_res_level,

        original_moving_file_path=original_moving_file_path,
        original_fixed_file_path=original_fixed_file_path,
        use_mirage=use_mirage,
    )
