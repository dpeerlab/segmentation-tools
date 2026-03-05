from loguru import logger
import numpy as np
from pathlib import Path
import tifffile
import skimage
from skimage.transform import ProjectiveTransform
from skimage.util import img_as_ubyte
import pyvips
from tqdm import tqdm
from segmentation_tools.utils import normalize, get_multiotsu_threshold
from segmentation_tools.utils.config import CHECKPOINT_DIR_NAME, RESULTS_DIR_NAME
from segmentation_tools.utils.profiling import profile_step, profile_block, log_array
import argparse


def _warp_channel_pyvips(
    channel_data: np.ndarray,
    combined_transform: np.ndarray,
    tile_height: int = 4096,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Warp a single channel using pyvips mapim with tiled processing for memory efficiency.

    Args:
        channel_data: 2D numpy array (H, W) with uint8 values.
        combined_transform: Dense coordinate map with shape (2, out_H, out_W),
            where [0] = row coords and [1] = col coords into source image.
        tile_height: Height of processing tiles.
        interpolation: Interpolation method ("bilinear", "bicubic", "nearest").

    Returns:
        Warped image as float32 array in [0, 1].
    """
    height, width = channel_data.shape
    out_rows, out_cols = combined_transform.shape[1], combined_transform.shape[2]

    # combined_transform is (2, H, W) where [0]=row, [1]=col
    # pyvips mapim expects (H, W, 2) with [x, y] i.e. [col, row]
    map_array = np.stack(
        [combined_transform[1], combined_transform[0]], axis=-1
    ).astype(np.float32)

    vips_img = pyvips.Image.new_from_memory(
        channel_data.tobytes(), width, height, 1, "uchar"
    )

    interp = pyvips.Interpolate.new(interpolation)

    tiles = []
    for start_row in tqdm(
        range(0, out_rows, tile_height), desc="Warping channel", leave=False
    ):
        end_row = min(start_row + tile_height, out_rows)
        tile_rows = end_row - start_row

        tile_coords = np.ascontiguousarray(map_array[start_row:end_row])
        map_img = pyvips.Image.new_from_memory(
            tile_coords.data, out_cols, tile_rows, 2, "float"
        )

        tile_warped = vips_img.mapim(map_img, interpolate=interp)
        tile_np = np.ndarray(
            buffer=tile_warped.write_to_memory(),
            dtype=np.uint8,
            shape=(tile_rows, out_cols),
        ).copy()
        tiles.append(tile_np)

    warped = np.concatenate(tiles, axis=0)
    return warped.astype(np.float32) / 255.0


def _apply_channel_warp_and_normalize(
    channel_data: np.ndarray,
    combined_transform: np.ndarray,
    tile_height: int = 4096,
    interpolation: str = "bilinear",
) -> np.ndarray:
    channel_normalized = normalize(channel_data)
    otsu_threshold_value = get_multiotsu_threshold(channel_normalized)

    logger.info(f"Otsu's threshold for channel: {otsu_threshold_value}")
    threshold = otsu_threshold_value
    channel_filtered = np.where(
        channel_normalized < threshold, 0, channel_normalized
    )

    # Convert to uint8 for pyvips
    channel_uint8 = img_as_ubyte(np.clip(channel_filtered, 0, 1).astype(np.float64))

    channel_warped = _warp_channel_pyvips(
        channel_uint8,
        combined_transform,
        tile_height=tile_height,
        interpolation=interpolation,
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

    logger.info(f"Moving image shape: {moving_image.shape}, transform shape: {combined_transform.shape}")
    if moving_image.ndim == 2:
        moving_image = moving_image[np.newaxis, ...]  # (H, W) -> (1, H, W)

    # 1. Apply Per-Channel Warp and Normalize to Level 0 Data
    logger.info("Applying per-channel normalization and warp to Level 0 (highest resolution) data.")
    
    # Use the original image data as the input for the Level 0 processing
    input_channels = [moving_image[:, :, i] for i in range(moving_image.shape[-1])]
    logger.info(f"Channel shape: {input_channels[0].shape}, transform shape: {combined_transform.shape}, n_channels: {len(input_channels)}")
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

    # 3. Save TIFF Pyramid (on CPU)
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

@profile_step("007 Warp All Channels and Downsample")
def main(
    moving_file_path: np.ndarray,
    high_res_level: int,
    linear_transform_file_path: Path,
    mirage_transform_file_path: Path,
    results_dir: Path,
):
    output_file_path = results_dir / "moving_complete_transform.ome.tiff"
    if output_file_path.exists():
        logger.info(f"Warped and downsampled file already exists at {output_file_path}. Skipping.")
        return 0

    with profile_block("Load moving image"):
        moving_image = tifffile.imread(
            moving_file_path, series=0, level=high_res_level, maxworkers=4
        )
        moving_image = np.moveaxis(np.array(moving_image), 0, -1)
    log_array("Moving image", moving_image)

    with profile_block("Load transforms"):
        linear_transform = np.load(linear_transform_file_path)
        mirage_warp = np.load(mirage_transform_file_path)
    log_array("Linear transform", linear_transform)
    log_array("MIRAGE warp", mirage_warp)

    with profile_block("Combine transforms"):
        combined_tranform = combine_transforms(
            mirage_warp=mirage_warp, linear_transform=linear_transform
        )

    n_levels = get_num_levels(moving_file_path) - high_res_level
    logger.info(f"Generating {n_levels}-level pyramid to {output_file_path}")

    with profile_block("Warp and save pyramidal TIFF"):
        warp_and_save_pyramidal_tiff(
            moving_image=moving_image,
            combined_transform=combined_tranform,
            n_levels=n_levels,
            output_file_path=output_file_path,
            description="Warped and Downsampled",
        )
    return 0

def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Warp all channels of a multi-resolution TIFF and downsample."
    )

    parser.add_argument(
        "--moving-file-path",
        required=True,
        type=str,
        help="Path to the moving TIFF file.",
    )
    parser.add_argument(
        "--high-res-level",
        required=True,
        type=int,
        help="High resolution level index.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    moving_file_path = args.moving_file_path
    high_res_level = args.high_res_level
    checkpoint_dir = Path(moving_file_path).parent.parent / CHECKPOINT_DIR_NAME

    linear_transform_file_path = checkpoint_dir / "linear_transform.npy"
    mirage_transform_file_path = checkpoint_dir / "mirage_transform.npy"
    results_dir = Path(moving_file_path).parent.parent / RESULTS_DIR_NAME

    main(
        moving_file_path=moving_file_path,
        high_res_level=high_res_level,
        linear_transform_file_path=linear_transform_file_path,
        mirage_transform_file_path=mirage_transform_file_path,
        results_dir=results_dir,
    )
