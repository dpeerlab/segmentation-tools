from icecream import ic
import sys
from loguru import logger
import numpy as np
from pathlib import Path
import tifffile
import skimage
from skimage.filters import threshold_multiotsu
from skimage.transform import ProjectiveTransform
from segmentation_tools.utils.image_utils import normalize



def _apply_channel_warp_and_normalize(
    channel_data: np.ndarray,
    combined_transform: np.ndarray,
) -> np.ndarray:
    channel_normalized = normalize(channel_data)
    otsu_threshold_value = get_multiotsu_threshold(channel_normalized)

    if otsu_threshold_value is None or otsu_threshold_value > 0.2:
        threshold = 0.15
        logger.warning(
            f"Otsu's threshold {otsu_threshold_value} is out of expected range [0.05, 0.5]. Using default threshold {threshold}."
        )
    else:
        logger.info(f"Otsu's threshold for channel: {otsu_threshold_value}")    
        threshold = otsu_threshold_value
        channel_filtered = np.where(
            channel_normalized < threshold, 0, channel_normalized
        )

    channel_warped = skimage.transform.warp(
        channel_filtered,
        inverse_map=combined_transform,
        output_shape=combined_transform.shape,
        order=3,
    )

    return channel_warped


def warp_and_save_pyramidal_tiff(
    moving_image: np.ndarray,
    combined_transform: np.ndarray,
    output_file_path: Path,
    n_levels: int,
    downscale_factor: int = 2,
    dtype_out: np.dtype = np.uint16,
    description: str = "",
):
    """
    Generates a multi-channel pyramid. Applies per-channel normalization and warp
    ONLY to the highest-resolution level (Level 0) data.
    """

    ic(moving_image.shape, combined_transform.shape)
    if moving_image.ndim == 2:
        moving_image = moving_image[np.newaxis, ...]  # (H, W) -> (1, H, W)

    # 1. Apply Per-Channel Warp and Normalize to Level 0 Data
    logger.info("Applying per-channel normalization and warp to Level 0 (highest resolution) data.")
    
    # Use the original image data as the input for the Level 0 processing
    input_channels = [moving_image[:, :, i] for i in range(moving_image.shape[-1])]
    # ic(input_channels)
    ic(input_channels[0].shape, combined_transform.shape, len(input_channels))
    processed_channels = []

    # Iterate over channels and apply both operations
    for i, ch in enumerate(input_channels):
        processed_ch = _apply_channel_warp_and_normalize(
            channel_data=ch, 
            combined_transform=combined_transform,
        )
        logger.info(f"Channel {i} processed.")
        processed_channels.append(processed_ch)
    
    # 'current' now holds the float [0, 1] processed data for Level 0
    current = np.stack(processed_channels, axis=0) 
    pyramid = []

    # 2. Pyramid Generation (Starts from Level 0 processed data)
    for i in range(n_levels):
        logger.info(f"Generating pyramid level {i}.")
        # Transfer from GPU (if applicable) to CPU (NumPy)
        current = current.get() if hasattr(current, 'get') else current 

        # Convert float [0, 1] to target integer dtype
        if dtype_out == np.uint8:
            # skimage scales float [0, 1] to uint8 [0, 255]
            current_int = skimage.util.img_as_ubyte(current)
        else:  # Default to uint16
            # skimage scales float [0, 1] to uint16 [0, 65535]
            current_int = skimage.util.img_as_uint(current)
        
        pyramid.append(current_int)

        if i < n_levels - 1:
            # Downsample for next level (GPU operation)
            new_h = max(1, current.shape[1] // downscale_factor)
            new_w = max(1, current.shape[2] // downscale_factor)

            # Downsample each channel separately
            current = np.stack(
                [
                    skimage.transform.resize(
                        ch, (new_h, new_w), preserve_range=True, anti_aliasing=True
                    )
                    for ch in current
                ],
                axis=0,
            )
            # Ensure the downsampled data remains in the [0, 1] range for the next loop
            current = np.clip(np.nan_to_num(current, nan=0.0), 0, 1)

    # 3. Save TIFF Pyramid (on CPU) - No change here
    with tifffile.TiffWriter(output_file_path, bigtiff=True) as tif:
        metadata = {
            "axes": "CYX",  # Indicates [Channel, Y, X] axes order
        }

        # Write Base Level (Level 0)
        tif.write(
            pyramid[0],
            subifds=len(pyramid) - 1,
            photometric="minisblack",
            metadata=metadata,
            dtype=dtype_out,
        )

        # Write Reduced-Resolution Levels (Levels 1 to N-1)
        for level_data in pyramid[1:]:
            tif.write(
                level_data, subfiletype=1, photometric="minisblack", dtype=dtype_out
            )

    logger.info(
        f"{description} multi-channel pyramidal TIFF saved to: {output_file_path}"
    )

def get_num_levels(tiff_file_path):
    """Get the number of resolution levels in a multi-resolution TIFF file."""
    with tifffile.TiffFile(tiff_file_path) as tif:
        return len(tif.series[0].levels)


def combine_transforms(
    mirage_warp: np.ndarray, linear_transform: np.ndarray
) -> np.ndarray:
    linear_transform = ProjectiveTransform(matrix=linear_transform)
    dense_coords_flat = mirage_warp.reshape(-1, 2)
    dense_coords_flat = dense_coords_flat[:, [1, 0]]
    final_coords_flat = linear_transform(dense_coords_flat)
    final_coords_flat = final_coords_flat[:, [1, 0]]
    final_coords_rc = final_coords_flat.reshape(mirage_warp.shape)

    final_inverse_map = final_coords_rc.transpose((2, 0, 1))
    return final_inverse_map

def main(
    moving_tiff_file_path: np.ndarray,
    high_res_level: int,
    linear_transform_file_path: Path,
    mirage_transform_file_path: Path,
    results_dir: Path,
):
    moving_image = tifffile.imread(
        moving_tiff_file_path, series=0, level=high_res_level, maxworkers=4
    )
    moving_image = np.moveaxis(
        np.array(moving_image), 0, -1
    )  # Move channel axis to end

    linear_transform = np.load(linear_transform_file_path)
    mirage_warp = np.load(mirage_transform_file_path)

    combined_tranform = combine_transforms(
        mirage_warp=mirage_warp, linear_transform=linear_transform
    )

    n_levels = get_num_levels(moving_tiff_file_path) - high_res_level
    _ = warp_and_save_pyramidal_tiff(
        moving_image=moving_image,
        combined_transform=combined_tranform,
        n_levels=n_levels,
        output_file_path=results_dir / f"moving_complete_transform.ome.tiff",
        description="Warped and Downsampled",
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error(
            "Usage: python 006_warp_all_channels_and_downsample.py <warped_file> <transform_file>"
        )
        sys.exit(1)

    moving_tiff_file_path = sys.argv[1]
    high_res_level = int(sys.argv[2])
    checkpoint_dir = Path(moving_tiff_file_path).parent.parent / ".checkpoints"

    linear_transform_file_path = checkpoint_dir / "linear_transform.npy"
    mirage_transform_file_path = checkpoint_dir / "mirage_transform.npy"
    results_dir = Path(moving_tiff_file_path).parent.parent / "results"

    main(
        moving_tiff_file_path=moving_tiff_file_path,
        high_res_level=high_res_level,
        linear_transform_file_path=linear_transform_file_path,
        mirage_transform_file_path=mirage_transform_file_path,
        results_dir=results_dir,
    )
