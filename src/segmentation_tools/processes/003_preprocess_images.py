from pathlib import Path
from loguru import logger
import numpy as np
import tifffile
import argparse

from segmentation_tools.utils import normalize, get_multiotsu_threshold
from segmentation_tools.utils.profiling import profile_step, profile_block, log_array

def load_image(input_file_path: Path, channel: int, level: int):
    """Load moving and fixed images from a multi-resolution OME-TIFF file."""
    with tifffile.TiffFile(input_file_path) as tif:
        if level >= len(tif.series[0].levels):
            raise ValueError(
                f"Requested level {level} exceeds available levels {len(tif.series[0].levels)-1}"
            )

    image = tifffile.imread(
        input_file_path,
        key=channel,
        series=0,
        level=level,
        maxworkers=4,
    )
    return image


@profile_step("003 Preprocess Images")
def main(input_file_path, dapi_channel_moving, level, output_file_path, filter):
    logger.info(f"Input: {input_file_path}, channel={dapi_channel_moving}, level={level}")
    logger.info(f"Output: {output_file_path}, filter={filter}")

    with profile_block("Load image"):
        dapi_image = load_image(
            input_file_path=input_file_path, channel=dapi_channel_moving, level=level
        )
    log_array("Loaded image", dapi_image)

    with profile_block("Normalize"):
        dapi_image_normalized = normalize(dapi_image)

    if not filter:
        np.save(output_file_path, dapi_image_normalized)
        logger.info(f"Saved normalized (unfiltered) to {output_file_path}")
        return

    with profile_block("Multi-Otsu threshold"):
        otsu_threshold_value = get_multiotsu_threshold(image=dapi_image_normalized)
    logger.info(f"Otsu's threshold: {otsu_threshold_value}")

    dapi_image_filtered = np.where(
        dapi_image_normalized < otsu_threshold_value, 0, dapi_image_normalized
    )

    np.save(output_file_path, dapi_image_filtered)
    logger.info(f"Saved filtered to {output_file_path}")
    return

def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Preprocess images by filtering DAPI channel using Otsu's method."
    )

    parser.add_argument(
        "--input-file-path",
        required=True,
        type=str,
        help="Path to the input OME-TIFF file.",
    )
    parser.add_argument(
        "--dapi-channel-moving",
        required=True,
        type=int,
        help="Channel index for the DAPI channel in the moving image.",
    )
    parser.add_argument(
        "--level",
        required=True,
        type=int,
        help="Pyramid level to load from the OME-TIFF file.",
    )
    parser.add_argument(
        "--output-file-path",
        required=True,
        type=str,
        help="Path to save the preprocessed output .npy file.",
    )
    parser.add_argument(
        "--filter",
        action='store_true',
        help="Whether to apply Otsu's filtering to the DAPI channel.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    input_file_path = args.input_file_path
    dapi_channel_moving = args.dapi_channel_moving
    level = args.level
    output_file_path = args.output_file_path
    filter = args.filter

    checkpoint_dir = Path(input_file_path).parent

    main(
        input_file_path=input_file_path,
        dapi_channel_moving=dapi_channel_moving,
        level=level,
        output_file_path=output_file_path,
        filter=filter,
    )
