from pathlib import Path
from loguru import logger
import sys
import numpy as np
import tifffile

from segmentation_tools.utils import normalize, get_multiotsu_threshold

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


def main(input_file_path, dapi_channel_moving, level, output_file_path, filter):
    dapi_image = load_image(
        input_file_path=input_file_path, channel=dapi_channel_moving, level=level
    )

    dapi_image_normalized = normalize(dapi_image)
    if not filter:
        np.save(output_file_path, dapi_image_normalized)
        return
    
    otsu_threshold_value = get_multiotsu_threshold(image=dapi_image_normalized, num_classes=4)
    logger.info(f"Otsu's threshold: {otsu_threshold_value}")

    dapi_image_filtered = np.where(
        dapi_image_normalized < otsu_threshold_value, 0, dapi_image_normalized
    )

    np.save(output_file_path, dapi_image_filtered)
    return


if __name__ == "__main__":
    # if len(sys.argv) != 4 and len(sys.argv) != 5:
    #     logger.error(
    #         "Usage: python preprocess_images.py <input_file_path> <dapi_channel_moving> <fixed_or_moving> [level]"
    #     )
    #     sys.exit(1)

    input_file_path = sys.argv[1]
    dapi_channel_moving = int(sys.argv[2])
    fixed_or_moving = sys.argv[3]

    filter = fixed_or_moving == "moving"

    checkpoint_dir = Path(input_file_path).parent
    if len(sys.argv) == 5:
        level = int(sys.argv[4])
    else:
        level_file_path = checkpoint_dir / "optimal_sift_level.txt"
        if not level_file_path.exists():
            raise FileNotFoundError(f"Optimal level file not found: {level_file_path}")

        with open(level_file_path, "r") as f:
            level = int(f.readline().strip())

    output_file_name = f"{fixed_or_moving}_dapi_filtered_level_{level}.npy"
    output_file_path = Path(input_file_path).parent / output_file_name

    main(
        input_file_path=input_file_path,
        dapi_channel_moving=dapi_channel_moving,
        level=level,
        output_file_path=output_file_path,
        filter=filter,
    )
