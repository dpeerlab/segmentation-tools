import sys
from loguru import logger
import numpy as np
from pathlib import Path
import tifffile
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from segmentation_tools.utils.profiling import profile_step, profile_block, log_array


@profile_step("010 SSIM Checks")
def main():
    checkpoint_dir = Path(sys.argv[1])

    with profile_block("Load pre-MIRAGE images"):
        moving = np.load(checkpoint_dir / "moving_dapi_linear_warped.npy")
        fixed = np.load(checkpoint_dir / "high_res_fixed_dapi_filtered_level_0.npy")
    log_array("Moving (linear warped)", moving)
    log_array("Fixed", fixed)

    with profile_block("Compute pre-MIRAGE SSIM"):
        mssim, ssim_full = ssim(moving, fixed, data_range=1.0, full=True)
    logger.info(f"Pre-MIRAGE SSIM: {mssim:.4f}")

    pre_mirage_ssim_path = checkpoint_dir / "ssim_pre_mirage.png"
    if pre_mirage_ssim_path.exists():
        logger.info(f"Pre-MIRAGE SSIM plot already exists at {pre_mirage_ssim_path}. Skipping.")
        return 0

    plt.imshow(ssim_full, cmap="gray")
    plt.suptitle(f"Pre-MIRAGE, SSIM: {mssim:.4f}")
    plt.savefig(pre_mirage_ssim_path)
    plt.close()

    results_dir = checkpoint_dir.parent / "results"
    mirage_warped_fp = results_dir / "moving_complete_transform.ome.tiff"

    with profile_block("Load post-MIRAGE DAPI"):
        mirage_dapi = tifffile.imread(mirage_warped_fp, series=0, level=0, key=1)
    log_array("Post-MIRAGE DAPI", mirage_dapi)

    if mirage_dapi.max() > 1.0:
        mirage_dapi = mirage_dapi / 65535.0

    with profile_block("Compute post-MIRAGE SSIM"):
        mssim, ssim_full = ssim(mirage_dapi, fixed, data_range=1.0, full=True)
    logger.info(f"Post-MIRAGE SSIM: {mssim:.4f}")

    plt.imshow(ssim_full, cmap="gray")
    plt.suptitle(f"Post-MIRAGE, SSIM: {mssim:.4f}")
    plt.savefig(checkpoint_dir / "ssim_post_mirage.png")



if __name__ == "__main__":
    main()

