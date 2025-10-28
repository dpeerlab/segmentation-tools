import skimage
import cv2
from loguru import logger
from pathlib import Path
import numpy as np
import tifffile
from skimage import img_as_ubyte
import os
import sys
from skimage.transform import ProjectiveTransform, AffineTransform, warp
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from icecream import ic


def get_SIFT_homography(
    image_fixed,  # Reference image to align to
    image_moving,  # Image to transform
    min_match_count: int = 10,
    save_image_path=None,
    draw_matches=False,
):
    FLANN_INDEX_KDTREE = 1

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # Find the keypoints and descriptors with SIFT
    kp_fixed, dsc_fixed = sift.detectAndCompute(image_fixed, None)
    kp_moving, dsc_moving = sift.detectAndCompute(image_moving, None)
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsc_fixed, dsc_moving, k=2)
    # Store all the good matches as per Lowe's ratio test.
    good_matches = list(filter(lambda m: m[0].distance < 0.7 * m[1].distance, matches))

    if save_image_path is not None:
        if draw_matches:
            image_matches = cv2.drawMatchesKnn(
                image_fixed,
                kp_fixed,
                image_moving,
                kp_moving,
                good_matches,
                None,
                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
            )
            cv2.imwrite(save_image_path, image_matches)
            logger.info(f"Saved SIFT matches to {save_image_path}")
        else:
            image_moving_resized = cv2.resize(
                image_moving, (image_fixed.shape[1], image_fixed.shape[0])
            )
            image_combined = np.hstack([image_fixed, image_moving_resized])

            # Save using matplotlib
            plt.figure(figsize=(10, 5))
            plt.imshow(image_combined, cmap="gray")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(save_image_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            logger.info(f"Saved side-by-side image (no matches) to {save_image_path}")

    # Compute 3x3 homography matrix
    if len(good_matches) >= min_match_count:
        fixed_pts = np.array([kp_fixed[m.queryIdx].pt for m, _ in good_matches])
        moving_pts = np.array([kp_moving[m.trainIdx].pt for m, _ in good_matches])
        M, mask = cv2.findHomography(moving_pts, fixed_pts, cv2.RANSAC, 5.0)
        return ProjectiveTransform(M)
    else:
        msg = (
            f"Insufficient no. matches detected to perform image "
            f"registration: {len(good_matches)} matches"
        )
        raise AssertionError(msg)


def _generate_image_variants(image: np.ndarray):
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


def _score_variant(
    variant_image,
    name,
    pre_matrix,
    fixed_image,
    save_image_dir=None,
    draw_matches=False,
):
    try:
        fixed_image = img_as_ubyte(fixed_image)
        variant_image = img_as_ubyte(variant_image)

        save_path = (
            f"{os.path.join(save_image_dir, name)}.png" if save_image_dir else None
        )
        H = get_SIFT_homography(
            image_fixed=fixed_image,
            image_moving=variant_image,
            save_image_path=save_path,
            draw_matches=draw_matches,
        )

        aligned = warp(
            variant_image,
            inverse_map=H.inverse,
            output_shape=fixed_image.shape,
            preserve_range=True,
        )

        score = ssim(fixed_image, aligned, data_range=aligned.max() - aligned.min())
        combined_matrix = H.params @ pre_matrix
        logger.info(f"Variant: {name}, Score: {score:.4f}")
        return (score, name, combined_matrix)

    except Exception as e:
        logger.warning(f"Failed to score variant {name}: {e}")
        return (-1, name, None)  # Failed variant


def find_best_sift(moving_image, fixed_image, save_image_dir=None, draw_matches=False):
    moving_image = img_as_ubyte(moving_image)
    fixed_image = img_as_ubyte(fixed_image)

    variants = _generate_image_variants(moving_image)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _score_variant,
                variant_image,
                name,
                pre_matrix,
                fixed_image,
                save_image_dir,
                draw_matches,
            )
            for variant_image, name, pre_matrix in variants
        ]

        results = [f.result() for f in futures]

    # Pick the best
    best = max(results, key=lambda r: r[0])  # r[0] = score
    best_score, best_name, best_matrix = best

    if best_matrix is None:
        raise RuntimeError("All variant alignments failed.")

    logger.info(f"Best alignment found: {best_name} with score {best_score:.4f}")
    return ProjectiveTransform(matrix=np.linalg.inv(best_matrix))


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


def main(
    moving_file_path,
    fixed_file_path,
    original_moving_file_path,
    original_fixed_file_path,
    high_res_level,
):
    checkpoint_dir = Path(moving_file_path).parent
    high_res_level = int(high_res_level)

    level_file_path = checkpoint_dir / "optimal_sift_level.txt"
    with open(level_file_path, "r") as f:
        level = f.read().strip()

    moving_level = int(level)
    fixed_level = int(level)

    ic("loading")
    moving_image_cp = np.load(moving_file_path)
    fixed_image_cp = np.load(fixed_file_path)

    ic("hist match")
    moving_image_cp = skimage.exposure.match_histograms(moving_image_cp, fixed_image_cp)

    ic("sift")
    sift_transform = find_best_sift(
        moving_image=moving_image_cp, fixed_image=fixed_image_cp
    )
    np.save(
        os.path.join(checkpoint_dir, "sift_transform.npy"),
        sift_transform.params,
    )

    ic("linear transform")
    ic("moving level", moving_level)
    ic("fixed level", fixed_level)
    ic("high res level", high_res_level)
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
    ic("saving linear transform")
    linear_output_path = checkpoint_dir / "linear_transform.npy"
    np.save(
        linear_output_path,
        linear_transform,
    )

    logger.info(f"Transformation linear map saved to {linear_output_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 6:
        logger.error(
            "Usage: python 003_find_alignment_transform.py <moving_file> <fixed_file> <high_res_level> <original_moving_file> <original_fixed_file>"
        )
        sys.exit(1)

    moving_file_path = sys.argv[1]
    fixed_file_path = sys.argv[2]
    high_res_level = sys.argv[3]
    original_moving_file_path = sys.argv[4]
    original_fixed_file_path = sys.argv[5]

    main(
        moving_file_path=moving_file_path,
        fixed_file_path=fixed_file_path,
        high_res_level=high_res_level,
        original_moving_file_path=original_moving_file_path,
        original_fixed_file_path=original_fixed_file_path,
    )
