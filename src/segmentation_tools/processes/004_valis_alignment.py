from pathlib import Path
import argparse
import itertools
import shutil
import warnings

import cv2
import numpy as np
import tifffile
from loguru import logger

from segmentation_tools.utils.profiling import profile_step, profile_block


def _extract_dapi_channel(tiff_path: Path, channel: int) -> np.ndarray:
    """Read a single channel from a (C, Y, X) or (Y, X) TIFF at full resolution."""
    with tifffile.TiffFile(str(tiff_path)) as tf:
        series = tf.series[0]
        axes = series.axes  # e.g. 'CYX', 'YX', 'TCYX', ...
        if 'C' not in axes:
            return series.asarray()
        c_idx = axes.index('C')
        arr = series.asarray()
        return np.take(arr, channel, axis=c_idx)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile-normalize to uint8 for feature detection."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _apply_orientation(img: np.ndarray, flip_x: bool, flip_y: bool, k_rot90: int) -> np.ndarray:
    if flip_x:
        img = np.fliplr(img)
    if flip_y:
        img = np.flipud(img)
    if k_rot90:
        img = np.rot90(img, k=k_rot90)
    return img


def _count_orb_inliers(img1_u8: np.ndarray, img2_u8: np.ndarray) -> int:
    """Count RANSAC inliers from ORB feature matching between two uint8 images."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, d1 = orb.detectAndCompute(img1_u8, None)
    kp2, d2 = orb.detectAndCompute(img2_u8, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 8:
        return 0
    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0
    return int(mask.sum())


def _detect_orientation(fixed_dapi: np.ndarray, moving_dapi: np.ndarray,
                        thumb_size: int = 512) -> tuple:
    """Find the best flip of moving_dapi to match fixed_dapi.

    Tests 4 flip combinations on thumbnails. Rotation is left to VALIS.
    Returns (flip_x, flip_y, k_rot90) where k_rot90 is always 0.
    """
    def _thumb(img):
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        return cv2.resize(_to_uint8(img), (nw, nh), interpolation=cv2.INTER_AREA)

    fixed_thumb = _thumb(fixed_dapi)
    moving_base = _thumb(moving_dapi)

    best_inliers = -1
    best_orientation = (False, False, 0)

    for flip_x, flip_y in itertools.product([False, True], [False, True]):
        candidate = _apply_orientation(moving_base.copy(), flip_x, flip_y, 0)
        inliers = _count_orb_inliers(fixed_thumb, candidate)
        logger.debug(f"  flip_x={flip_x} flip_y={flip_y}: {inliers} inliers")
        if inliers > best_inliers:
            best_inliers = inliers
            best_orientation = (flip_x, flip_y, 0)

    flip_x, flip_y, _ = best_orientation
    logger.info(f"Best flip: flip_x={flip_x}, flip_y={flip_y} ({best_inliers} inliers)")
    return best_orientation


def _orientation_to_matrix(flip_x: bool, flip_y: bool, k_rot90: int,
                            shape_rc: tuple) -> np.ndarray:
    """Return the 3×3 homogeneous matrix for the pre-orientation transform.

    This is the matrix that maps coordinates in the oriented (VALIS-input) image
    back to coordinates in the original moving image, so it can be composed with
    the VALIS transform when building the full inverse map.
    """
    H, W = shape_rc

    # Build forward transform: original → oriented
    # We apply: flip_x, then flip_y, then rot90 k times
    # Represent each as a 3×3 matrix acting on (x, y, 1) column vectors

    def flip_x_M(w):
        return np.array([[-1, 0, w - 1],
                         [ 0, 1, 0],
                         [ 0, 0, 1]], dtype=np.float64)

    def flip_y_M(h):
        return np.array([[1,  0, 0],
                         [0, -1, h - 1],
                         [0,  0, 1]], dtype=np.float64)

    def rot90_M(k, h, w):
        """Rotation by k*90° CCW around image centre, mapping to a new canvas."""
        k = k % 4
        if k == 0:
            return np.eye(3, dtype=np.float64)
        if k == 1:  # 90° CCW: (x,y) -> (y, W-1-x), output size (W, H)
            return np.array([[0, 1, 0],
                             [-1, 0, w - 1],
                             [0,  0, 1]], dtype=np.float64)
        if k == 2:  # 180°
            return np.array([[-1, 0, w - 1],
                             [0, -1, h - 1],
                             [0,  0, 1]], dtype=np.float64)
        if k == 3:  # 270° CCW (= 90° CW)
            return np.array([[0, -1, h - 1],
                             [1,  0, 0],
                             [0,  0, 1]], dtype=np.float64)

    M = np.eye(3, dtype=np.float64)
    cur_h, cur_w = H, W

    if flip_x:
        M = flip_x_M(cur_w) @ M
        # flip_x doesn't change shape
    if flip_y:
        M = flip_y_M(cur_h) @ M

    if k_rot90 % 2 == 1:
        # 90° or 270° swaps dimensions
        rot_M = rot90_M(k_rot90, cur_h, cur_w)
        M = rot_M @ M
        cur_h, cur_w = cur_w, cur_h
    elif k_rot90 in (2,):
        M = rot90_M(k_rot90, cur_h, cur_w) @ M

    return M  # forward: original coords → oriented coords


def _scale_matrix_to_full_res(M, processed_shape_rc, full_shape_rc):
    """Scale a transform matrix from processed resolution to full resolution."""
    sy = full_shape_rc[0] / processed_shape_rc[0]
    sx = full_shape_rc[1] / processed_shape_rc[1]
    S_up = np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0,  0, 1]], dtype=np.float64)
    S_down = np.linalg.inv(S_up)
    return S_up @ M @ S_down


@profile_step("004 VALIS Alignment")
def main(fixed_file_path, moving_file_path, fixed_dapi_channel,
         moving_dapi_channel, checkpoint_dir):
    """Run VALIS rigid/affine registration and save the linear transform.

    Outputs
    -------
    checkpoint_dir / "linear_transform.npy"
        3×3 inverse map matrix (fixed-space -> moving-space), compatible with
        ``skimage.transform.warp`` and step 007's ``combine_transforms()``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    fixed_file_path = Path(fixed_file_path)
    moving_file_path = Path(moving_file_path)

    output_path = checkpoint_dir / "linear_transform.npy"
    if output_path.exists():
        logger.info(f"Linear transform already exists at {output_path}. Skipping.")
        return 0

    # --- 1. Prepare VALIS input/output directories ---
    valis_input_dir = checkpoint_dir / "valis_input"
    valis_output_dir = checkpoint_dir / "valis_output"
    if valis_output_dir.exists():
        shutil.rmtree(valis_output_dir)
    if valis_input_dir.exists():
        shutil.rmtree(valis_input_dir)
    valis_input_dir.mkdir()
    valis_output_dir.mkdir()

    # --- 2. Extract DAPI channels ---
    with profile_block("Extract DAPI channels"):
        fixed_dapi = _extract_dapi_channel(fixed_file_path, fixed_dapi_channel)
        moving_dapi = _extract_dapi_channel(moving_file_path, moving_dapi_channel)

    fixed_full_shape_rc = fixed_dapi.shape[-2:]
    moving_full_shape_rc = moving_dapi.shape[-2:]

    logger.info(f"Fixed DAPI shape: {fixed_full_shape_rc}, dtype: {fixed_dapi.dtype}")
    logger.info(f"Moving DAPI shape: {moving_full_shape_rc}, dtype: {moving_dapi.dtype}")

    # --- 3. Detect orientation and pre-orient moving image ---
    with profile_block("Detect orientation"):
        flip_x, flip_y, k_rot90 = _detect_orientation(fixed_dapi, moving_dapi)

    oriented_moving = _apply_orientation(moving_dapi, flip_x, flip_y, k_rot90)
    oriented_shape_rc = oriented_moving.shape[:2]
    logger.info(f"Oriented moving shape: {oriented_shape_rc}")

    # Build the orientation matrix (original moving coords → oriented moving coords)
    orient_M_fwd = _orientation_to_matrix(flip_x, flip_y, k_rot90, moving_full_shape_rc)

    # --- 4. Write flat TIFFs for VALIS ---
    fixed_staged = valis_input_dir / "00_fixed.tif"
    moving_staged = valis_input_dir / "01_moving.tif"

    with profile_block("Write flat DAPI TIFFs"):
        tifffile.imwrite(str(fixed_staged), fixed_dapi)
        tifffile.imwrite(str(moving_staged), oriented_moving)

    logger.info(f"Wrote {fixed_staged.name} and {moving_staged.name} to {valis_input_dir}")
    del fixed_dapi, moving_dapi, oriented_moving

    # --- 5. Run VALIS registration ---
    from valis import registration, micro_rigid_registrar as mrr

    with profile_block("VALIS registration"):
        registrar = registration.Valis(
            src_dir=str(valis_input_dir),
            dst_dir=str(checkpoint_dir),
            name="valis_output",
            reference_img_f="00_fixed.tif",
            align_to_reference=True,
            image_type="multi",
            crop=registration.CROP_REF,
            micro_rigid_registrar_cls=mrr.MicroRigidRegistrar,
            micro_rigid_registrar_params={
                "scale": 0.5**2,
                "tile_wh": 512,
                "roi": "mask",
            },
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="valis")
            warnings.filterwarnings("ignore", category=UserWarning, module="valis")
            try:
                _, _, error_df = registrar.register()
            except AttributeError as e:
                if "non_rigid_reg_kwargs" in str(e):
                    logger.warning(f"VALIS cleanup error (non-fatal): {e}")
                    error_df = None
                else:
                    raise

    logger.info("VALIS registration complete")
    if error_df is not None:
        logger.info(f"Registration error summary:\n{error_df.to_string()}")

    # --- 6. Extract VALIS transform ---
    moving_stem = moving_staged.stem  # "01_moving"
    moving_slide = registrar.slide_dict.get(moving_stem)
    if moving_slide is None:
        for key, slide in registrar.slide_dict.items():
            if "moving" in key.lower():
                moving_slide = slide
                break

    if moving_slide is None:
        raise RuntimeError(
            f"Could not find moving slide in VALIS results. "
            f"Available keys: {list(registrar.slide_dict.keys())}"
        )

    forward_M = moving_slide.M
    if forward_M is None:
        raise RuntimeError(
            "VALIS registration failed: slide.M is None. "
            "Check that the DAPI channels are correct and the images overlap."
        )
    logger.info(f"VALIS forward transform (processed res):\n{forward_M}")

    processed_shape_rc = moving_slide.processed_img_shape_rc
    logger.info(f"Processed shape (VALIS): {processed_shape_rc}")
    logger.info(f"Oriented moving shape: {oriented_shape_rc}")
    logger.info(f"Original moving shape: {moving_full_shape_rc}")

    # --- 7. Compose full transform and invert ---
    # VALIS gives: oriented_moving_processed → fixed_processed (forward)
    # We need: fixed_full → original_moving_full (inverse map for skimage.warp)
    #
    # Chain:
    #   fixed_full → fixed_processed  (S_down_fixed)
    #   → oriented_moving_processed    (VALIS forward_M ... but VALIS M maps
    #                                   processed moving → processed fixed,
    #                                   so inverse of forward_M maps fixed → moving)
    #   → oriented_moving_full         (S_up_oriented)
    #   → original_moving_full         (inv(orient_M_fwd))
    #
    # Simpler: scale VALIS forward M to oriented full-res, then compose with orient_M_fwd
    with profile_block("Compose and invert transform"):
        forward_M_oriented_full = _scale_matrix_to_full_res(
            forward_M, processed_shape_rc, oriented_shape_rc
        )
        logger.info(f"VALIS forward transform (oriented full res):\n{forward_M_oriented_full}")

        # Full forward map: original_moving → fixed
        #   = forward_M_oriented_full @ orient_M_fwd
        full_forward_M = forward_M_oriented_full @ orient_M_fwd
        logger.info(f"Full forward transform (original moving → fixed):\n{full_forward_M}")

        # Invert: downstream steps use inverse map (fixed → original moving)
        inverse_M = np.linalg.inv(full_forward_M)

    logger.info(f"Inverse transform (linear_transform.npy):\n{inverse_M}")

    np.save(output_path, inverse_M)
    logger.info(f"Linear transform saved to {output_path}")
    return 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run VALIS rigid/affine alignment on fixed and moving images."
    )
    parser.add_argument("--fixed-file-path", required=True, type=str)
    parser.add_argument("--moving-file-path", required=True, type=str)
    parser.add_argument("--fixed-dapi-channel", type=int, default=0)
    parser.add_argument("--moving-dapi-channel", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    checkpoint_dir = Path(args.fixed_file_path).parent
    main(
        fixed_file_path=args.fixed_file_path,
        moving_file_path=args.moving_file_path,
        fixed_dapi_channel=args.fixed_dapi_channel,
        moving_dapi_channel=args.moving_dapi_channel,
        checkpoint_dir=checkpoint_dir,
    )
