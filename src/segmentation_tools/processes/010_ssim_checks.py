import sys
from loguru import logger
import numpy as np
from pathlib import Path
import tifffile
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    checkpoint_dir = sys.argv[1]
    moving = np.load(
        os.path.join(checkpoint_dir, "moving_dapi_linear_warped.npy")
    )

    fixed = np.load(
        os.path.join(checkpoint_dir, "high_res_fixed_dapi_filtered_level_0.npy")
    )

    mssim, ssim_full = ssim(
        moving,
        fixed,
        data_range=1.0,
        full=True
    )

    plt.imshow(ssim_full, cmap = "gray")
    plt.suptitle(f"Pre-MIRAGE, SSIM: {mssim}")
    plt.savefig(os.path.join(checkpoint_dir, "ssim_pre_mirage.png"))
    plt.close()

    results_dir = Path(checkpoint_dir).parent / "results"

    mirage_warped_fp = results_dir /  "moving_complete_transform.ome.tiff"

    mirage_dapi = tifffile.imread(
        mirage_warped_fp,
        series = 0,
        level = 0,
        key = 1
    )

    if mirage_dapi.max() > 1.0:
        mirage_dapi = mirage_dapi / 65535.0

    mssim, ssim_full = ssim(
        mirage_dapi,
        fixed,
        data_range=1.0,
        full=True
    )

    plt.imshow(ssim_full, cmap = "gray")
    plt.suptitle(f"Post-MIRAGE, SSIM: {mssim}")
    plt.savefig(os.path.join(checkpoint_dir, "ssim_post_mirage.png"))


