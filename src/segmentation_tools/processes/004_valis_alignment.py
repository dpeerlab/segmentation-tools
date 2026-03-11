"""VALIS-based rigid/affine alignment of moving image to fixed reference.

Replaces the multi-step SIFT pipeline with a single VALIS registration call.
Produces ``linear_transform.npy`` (3×3 inverse map matrix) compatible with
downstream steps 005, 007.

Design decisions
----------------
SimilarityTransform (VALIS default) is the right transform class:
- EuclideanTransform (rotation + translation) is too rigid — doesn't handle
  the scale difference between Xenium and IF (different pixel sizes).
- SimilarityTransform (rotation + uniform scale + translation) is correct.
  Xenium and IF image the same physical tissue, so any scale difference is
  uniform across the field of view.
- AffineTransform adds independent x/y scale and shear, which are physically
  unmotivated for flat tissue imaged by two modalities.
- ProjectiveTransform adds perspective distortion, which is wrong here.

VALIS reads OME-TIFF natively, but its reader selection logic always prefers
pyvips when the file extension is recognised (e.g. .tiff). pyvips silently
returns zeros for JPEG2000-compressed TIFFs (compression code 34712), which
the Xenium morphology image uses. We detect JPEG2000 files and pass them to
VALIS via reader_dict with BioFormatsSlideReader, which handles JPEG2000
correctly and benefits from the pyramid for fast level-of-detail reads.
All files are symlinked into a staging directory (zero I/O cost).

check_for_reflections=True tests all 3 non-trivial flip variants (x-flip,
y-flip, xy-flip) and picks the one with lowest registration error. This
handles the horizontal flip between Xenium and IF automatically.
The default VALIS affine optimizer (AffineOptimizerMattesMI) is used.
"""

from pathlib import Path
import argparse
import os
import shutil
import warnings

import numpy as np
import tifffile
from loguru import logger

from segmentation_tools.utils.profiling import profile_step, profile_block

# pyvips silently returns zeros for JPEG2000-compressed TIFFs.
_JPEG2000_COMPRESSION = 34712


def _is_jpeg2000(tiff_path):
    """Return True if the TIFF uses JPEG2000 compression (pyvips-incompatible)."""
    with tifffile.TiffFile(str(tiff_path)) as tf:
        return tf.pages[0].compression == _JPEG2000_COMPRESSION


def _scale_matrix_to_full_res(M, processed_shape_rc, full_shape_rc):
    """Scale a transform matrix from processed resolution to full resolution.

    VALIS computes M at a downsampled (processed) resolution. To apply it at
    full resolution we conjugate with scaling matrices:

        M_full = S_up @ M_proc @ S_down

    Parameters
    ----------
    M : np.ndarray (3, 3)
    processed_shape_rc : tuple  (rows, cols) at processed resolution
    full_shape_rc : tuple  (rows, cols) at full resolution
    """
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
    # Always wipe valis_output to avoid VALIS getting confused by stale state
    # from previous failed runs (it warns and misbehaves if n_images changes).
    valis_input_dir = checkpoint_dir / "valis_input"
    valis_output_dir = checkpoint_dir / "valis_output"
    if valis_output_dir.exists():
        shutil.rmtree(valis_output_dir)
    if valis_input_dir.exists():
        shutil.rmtree(valis_input_dir)
    valis_input_dir.mkdir()
    valis_output_dir.mkdir()

    # Symlink originals with ordered names so VALIS sees them in the right
    # sequence (it sorts filenames). Preserve the suffix so VALIS picks the
    # right format reader as a fallback.
    fixed_suffix = "".join(fixed_file_path.suffixes)
    moving_suffix = "".join(moving_file_path.suffixes)
    fixed_staged = valis_input_dir / f"00_fixed{fixed_suffix}"
    moving_staged = valis_input_dir / f"01_moving{moving_suffix}"
    os.symlink(fixed_file_path.resolve(), fixed_staged)
    os.symlink(moving_file_path.resolve(), moving_staged)
    logger.info(f"Staged {fixed_file_path.name} (DAPI ch={fixed_dapi_channel}) -> {fixed_staged.name}")
    logger.info(f"Staged {moving_file_path.name} (DAPI ch={moving_dapi_channel}) -> {moving_staged.name}")

    # --- 2. Run VALIS registration ---
    from valis import registration, slide_io

    # VALIS prefers pyvips based purely on file extension, without checking
    # compression. Build a reader_dict to force BioFormatsSlideReader for any
    # JPEG2000-compressed file (pyvips silently returns zeros for those).
    # reader_dict values are (reader_class, kwargs) tuples per the VALIS API.
    slide_io.init_jvm()
    reader_dict = {}
    for staged, src in [(fixed_staged, fixed_file_path), (moving_staged, moving_file_path)]:
        if _is_jpeg2000(src):
            logger.info(
                f"{src.name} uses JPEG2000 compression — using BioFormatsSlideReader"
            )
            reader_dict[str(staged)] = (slide_io.BioFormatsSlideReader, {})

    # Channel selection: VALIS applies if_processing_kwargs to all images.
    # The fixed is single-channel so the channel param is ignored for it.
    if_kwargs = {"channel": moving_dapi_channel, "adaptive_eq": True}

    with profile_block("VALIS registration"):
        registrar = registration.Valis(
            src_dir=str(valis_input_dir),
            dst_dir=str(checkpoint_dir),
            name="valis_output",
            imgs_ordered=True,
            reference_img_f=fixed_staged.name,
            align_to_reference=True,
            non_rigid_registrar_cls=None,       # MIRAGE handles non-rigid
            image_type="fluorescence",
            check_for_reflections=True,         # Detect x/y/xy flips automatically
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="valis")
            warnings.filterwarnings("ignore", category=UserWarning, module="valis")
            try:
                _, _, error_df = registrar.register(
                    reader_dict=reader_dict or None,
                    if_processing_kwargs=if_kwargs,
                )
            except AttributeError as e:
                if "non_rigid_reg_kwargs" in str(e):
                    logger.warning(f"VALIS cleanup error (non-fatal): {e}")
                    error_df = None
                else:
                    raise

    logger.info("VALIS registration complete")
    if error_df is not None:
        logger.info(f"Registration error summary:\n{error_df.to_string()}")

    # --- 3. Extract transform ---
    # Slide dict keys are the filename stem (first dot-separated part).
    moving_stem = moving_staged.name.split(".")[0]   # "01_moving"
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
    full_shape_rc = moving_slide.slide_shape_rc
    logger.info(f"Processed shape: {processed_shape_rc}, Full shape: {full_shape_rc}")

    with profile_block("Scale and invert transform"):
        forward_M_full = _scale_matrix_to_full_res(
            forward_M, processed_shape_rc, full_shape_rc
        )
        logger.info(f"VALIS forward transform (full res):\n{forward_M_full}")
        # Invert: downstream steps expect inverse map (fixed -> moving)
        inverse_M = np.linalg.inv(forward_M_full)

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
