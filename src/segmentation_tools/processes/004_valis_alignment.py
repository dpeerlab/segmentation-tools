"""VALIS-based rigid/affine alignment of moving image to fixed reference.

Replaces the multi-step SIFT pipeline (steps 002-004) with a single VALIS
registration call. Produces ``linear_transform.npy`` (3×3 inverse map matrix)
compatible with downstream steps 005, 007.

VALIS handles multi-resolution feature detection, matching, and affine
optimization automatically. Non-rigid registration is disabled here because
MIRAGE (step 006) handles that.
"""

from pathlib import Path
import argparse
import shutil

import numpy as np
import tifffile
from loguru import logger
from skimage.transform import AffineTransform

from segmentation_tools.utils.profiling import profile_step, profile_block, log_array


def _extract_dapi_channel(tiff_path, channel_idx, output_path):
    """Extract a single DAPI channel from a multi-channel TIFF and save it.

    Parameters
    ----------
    tiff_path : Path
        Path to multi-channel OME-TIFF.
    channel_idx : int
        Channel index for DAPI.
    output_path : Path
        Where to write the single-channel TIFF.

    Returns
    -------
    np.ndarray
        The extracted channel data.
    """
    img = tifffile.imread(str(tiff_path), series=0, level=0, key=channel_idx)
    tifffile.imwrite(str(output_path), img, compression="zlib")
    logger.info(f"Extracted channel {channel_idx} from {tiff_path.name} -> {output_path.name}")
    log_array("Extracted channel", img)
    return img


def _scale_matrix_to_full_res(M, processed_shape_rc, full_shape_rc):
    """Scale a transform matrix from processed resolution to full resolution.

    VALIS computes M at a downsampled (processed) resolution. To apply it at
    full resolution, we conjugate with scaling matrices:

        M_full = S_up @ M_proc @ S_down

    where S_up scales from processed to full, and S_down scales from full to
    processed.

    Parameters
    ----------
    M : np.ndarray (3, 3)
        Transform matrix at processed resolution.
    processed_shape_rc : tuple
        (rows, cols) of the processed image.
    full_shape_rc : tuple
        (rows, cols) of the full-resolution image.

    Returns
    -------
    np.ndarray (3, 3)
        Transform matrix at full resolution.
    """
    sy = full_shape_rc[0] / processed_shape_rc[0]
    sx = full_shape_rc[1] / processed_shape_rc[1]

    # Scale up: processed -> full res
    S_up = np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0,  0, 1]], dtype=np.float64)

    # Scale down: full res -> processed
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

    # --- 1. Prepare VALIS input directory with single-channel DAPI images ---
    valis_input_dir = checkpoint_dir / "valis_input"
    valis_output_dir = checkpoint_dir / "valis_output"
    valis_input_dir.mkdir(exist_ok=True)
    valis_output_dir.mkdir(exist_ok=True)

    with profile_block("Extract DAPI channels"):
        # Prefix with 00_/01_ to ensure VALIS ordering (fixed first)
        fixed_dapi_path = valis_input_dir / "00_fixed_dapi.tiff"
        moving_dapi_path = valis_input_dir / "01_moving_dapi.tiff"

        _extract_dapi_channel(fixed_file_path, fixed_dapi_channel, fixed_dapi_path)
        _extract_dapi_channel(moving_file_path, moving_dapi_channel, moving_dapi_path)

    # --- 2. Run VALIS registration (rigid/affine only) ---
    # Import here to keep valis an optional dependency
    from valis import registration

    with profile_block("VALIS registration"):
        registrar = registration.Valis(
            str(valis_input_dir),
            str(valis_output_dir),
            imgs_ordered=True,
            reference_img_f="00_fixed_dapi.tiff",
            align_to_reference=True,
            non_rigid_registrar_cls=None,  # Skip non-rigid; MIRAGE handles this
            image_type="fluorescence",
        )
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    logger.info(f"VALIS registration complete")
    if error_df is not None:
        logger.info(f"Registration error summary:\n{error_df.to_string()}")

    # --- 3. Extract transform for the moving slide ---
    # VALIS keys slide_dict by filename stem (without extension)
    moving_slide = registrar.slide_dict.get("01_moving_dapi")
    if moving_slide is None:
        # Fallback: try with extension
        for key, slide in registrar.slide_dict.items():
            if "moving" in key.lower():
                moving_slide = slide
                break

    if moving_slide is None:
        raise RuntimeError(
            f"Could not find moving slide in VALIS results. "
            f"Available keys: {list(registrar.slide_dict.keys())}"
        )

    # VALIS M is a forward transform (moving -> fixed) at processed resolution
    forward_M = moving_slide.M
    logger.info(f"VALIS forward transform (processed res):\n{forward_M}")

    processed_shape_rc = moving_slide.processed_img_shape_rc
    full_shape_rc = moving_slide.slide_shape_rc
    logger.info(f"Processed shape: {processed_shape_rc}, Full shape: {full_shape_rc}")

    # Scale M to full resolution
    with profile_block("Scale and invert transform"):
        forward_M_full = _scale_matrix_to_full_res(
            forward_M, processed_shape_rc, full_shape_rc
        )
        logger.info(f"VALIS forward transform (full res):\n{forward_M_full}")

        # Invert: step 005 and 007 expect the inverse map (fixed -> moving)
        # This is the convention used by skimage.transform.warp
        inverse_M = np.linalg.inv(forward_M_full)

    logger.info(f"Inverse transform (linear_transform.npy):\n{inverse_M}")

    # --- 4. Save ---
    np.save(output_path, inverse_M)
    logger.info(f"Linear transform saved to {output_path}")

    # --- 5. Cleanup ---
    registrar.close()

    # Optionally clean up valis working directories (keep input for debugging)
    # shutil.rmtree(valis_output_dir, ignore_errors=True)

    return 0


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VALIS rigid/affine alignment on fixed and moving images."
    )
    parser.add_argument(
        "--fixed-file-path", required=True, type=str,
        help="Path to the fixed TIFF file.",
    )
    parser.add_argument(
        "--moving-file-path", required=True, type=str,
        help="Path to the moving TIFF file.",
    )
    parser.add_argument(
        "--fixed-dapi-channel", type=int, default=0,
        help="DAPI channel index in the fixed image (default: 0).",
    )
    parser.add_argument(
        "--moving-dapi-channel", type=int, default=1,
        help="DAPI channel index in the moving image (default: 1).",
    )
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
