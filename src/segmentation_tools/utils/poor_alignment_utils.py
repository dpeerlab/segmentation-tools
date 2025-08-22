import numpy as np
import cupy as cp
import tqdm
from cucim.skimage.metrics import structural_similarity as cucim_ssim
import matplotlib.pyplot as plt
from typing import Tuple
import os

def compute_ssim_tile_heatmap(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    threshold: float = 0.1,
    save_img_dir: os.PathLike = None,
    prefix: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a full-resolution SSIM map and an upsampled per-tile SSIM heatmap.

    Returns:
        full_ssim_map (np.ndarray): (H, W) SSIM map per pixel.
        heatmap_upsampled (np.ndarray): (H, W) map of per-tile mean SSIM scores, upsampled via nearest-neighbor.
    """
    if type(fixed_image) is np.ndarray:
        fixed_image_gpu = cp.asarray(fixed_image, dtype=cp.float32)
        moving_img_gpu = cp.asarray(moving_image, dtype=cp.float32)
    else:
        fixed_image_gpu = fixed_image
        moving_img_gpu = moving_image

    fixed_image_gpu = cp.where(fixed_image_gpu > threshold, fixed_image_gpu, 0)
    moving_img_gpu = cp.where(moving_img_gpu > threshold, moving_img_gpu, 0)

    rows, cols = fixed_image_gpu.shape
    full_ssim_map = cp.zeros_like(fixed_image_gpu, dtype=cp.float32)

    step_y, step_x = tile_size - overlap, tile_size - overlap
    total_tiles_y = (rows + step_y - 1) // step_y
    total_tiles_x = (cols + step_x - 1) // step_x

    tile_heatmap_gpu = cp.zeros((total_tiles_y, total_tiles_x), dtype=cp.float32)

    pbar = tqdm.tqdm(total=total_tiles_y * total_tiles_x, desc="Calculating SSIM Tiles")
    for tile_y, i in enumerate(range(0, rows, step_y)):
        for tile_x, j in enumerate(range(0, cols, step_x)):
            r_min, r_max = i, min(i + tile_size, rows)
            c_min, c_max = j, min(j + tile_size, cols)

            ref_tile = fixed_image_gpu[r_min:r_max, c_min:c_max]
            moving_tile = moving_img_gpu[r_min:r_max, c_min:c_max]

            try:
                score, ssim_map_tile = cucim_ssim(
                    ref_tile, moving_tile, full=True, data_range=1.0
                )
                full_ssim_map[r_min:r_max, c_min:c_max] = ssim_map_tile
                tile_heatmap_gpu[tile_y, tile_x] = score
                # tile_heatmap_normalized_gpu = tile_heatmap_gpu[tile_y, tile_x] / cp.sum(ref_tile + moving_tile)
            except Exception as e:
                print(f"Error processing tile at ({i}, {j}): {e}")
                full_ssim_map[r_min:r_max, c_min:c_max] = 0.0
                tile_heatmap_gpu[tile_y, tile_x] = 0.0
                # tile_heatmap_normalized_gpu[tile_y, tile_x] = 0.0

            pbar.update(1)
    pbar.close()

    # Nearest-neighbor upsample on GPU
    heatmap_upsampled_gpu = cp.repeat(cp.repeat(tile_heatmap_gpu, step_y, axis=0), step_x, axis=1)
    heatmap_upsampled_gpu = heatmap_upsampled_gpu[:rows, :cols]

    # heatmap_upsampled_normalized_gpu = cp.repeat(cp.repeat(tile_heatmap_normalized_gpu, step_y, axis=0), step_x, axis=1)
    # heatmap_upsampled_normalized_gpu = heatmap_upsampled_normalized_gpu[:rows, :cols]

    # Move to CPU
    full_ssim_map_cpu = cp.asnumpy(full_ssim_map)
    heatmap_upsampled_cpu = cp.asnumpy(heatmap_upsampled_gpu)
    # heatmap_upsampled_normalized_cpu = cp.asnumpy(heatmap_upsampled_normalized_gpu)

    if save_img_dir:
        ssim_full_name = "ssim_full.png"
        ssim_heatmap_name = "ssim_heatmap.png"

        if prefix:
            ssim_full_name = f"{prefix}_{ssim_full_name}"
            ssim_heatmap_name = f"{prefix}_{ssim_heatmap_name}"

        plt.imsave(os.path.join(save_img_dir, ssim_full_name), full_ssim_map_cpu, cmap="gray")
        plt.imsave(os.path.join(save_img_dir, ssim_heatmap_name), heatmap_upsampled_cpu, cmap="hot")


    return full_ssim_map_cpu, heatmap_upsampled_cpu,
