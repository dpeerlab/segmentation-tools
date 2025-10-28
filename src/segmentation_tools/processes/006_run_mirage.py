import sys
from loguru import logger
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
import mirage


def run_mirage(
    warped_image,
    fixed_image,
    bin_mask=None,
    pad=12,
    offset=12,
    num_neurons=300,
    num_layers=3,
    pool=1,
    loss="SSIM",
    batch_size=256,
    num_steps=2048,
    lr=0.001,
):
    """
    Run MIRAGE alignment on the given images.

    Parameters:
        warped_image: The warped image to align.
        fixed_image: The fixed image to align against.
        bin_mask: Optional binary mask for the warped image.
        pad: Padding for the model.
        offset: Offset for the model.
        num_neurons: Number of neurons in the model.
        num_layers: Number of layers in the model.
        pool: Pooling factor for the model.
        loss: Loss function to use ("SSIM" or "MSE").
        save_img_dir: Directory to save aligned images (optional).

    Returns:
        Aligned image as a numpy array.
    """
    logger.info("Running MIRAGE alignment...")

    mirage_model = mirage.MIRAGE(
        images=warped_image,
        references=fixed_image,
        bin_mask=bin_mask,
        pad=pad,
        offset=offset,
        num_neurons=num_neurons,
        num_layers=num_layers,
        pool=pool,
        loss=loss,
    )

    logger.info("Training MIRAGE model...")
    try:
        mirage_model.train(batch_size=256, num_steps=2048, lr__sched=True, LR=0.001)

        logger.info("MIRAGE model training complete. Computing transform...")

        mirage_model.compute_transform()
    except Exception as e:
        logger.error(f"Error computing transform: {e}")
        return None

    return mirage_model.get_transform()


def main(warped_file_path, fixed_file_path, checkpoint_dir):
    warped_image = np.load(warped_file_path)
    mirage_transform = run_mirage(
        warped_image=warped_image,
        fixed_image=np.load(fixed_file_path),
        pad=15,
        offset=15,
        num_neurons=400,
        num_layers=4,
        pool=1,
        loss="SSIM",
    )

    if mirage_transform is None:
        logger.error("MIRAGE alignment failed.")
        return 1
    mirage_transform_file_path = checkpoint_dir / "mirage_transform.npy"
    np.save(mirage_transform_file_path, mirage_transform)
    logger.info(f"MIRAGE transform saved to {mirage_transform_file_path}")
    return 0


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     logger.error(
    #         "Usage: python 004_warp_image_with_sift.py <warped_file> <transform_file>"
    #     )
    #     sys.exit(1)

    warped_file_path = sys.argv[1]
    fixed_file_path = sys.argv[2]
    batch_size = sys.argv[3] if len(sys.argv) > 3 else 256
    lr = sys.argv[4] if len(sys.argv) > 4 else 0.001

    checkpoint_dir = Path(warped_file_path).parent
    main(
        warped_file_path=warped_file_path,
        fixed_file_path=fixed_file_path,
        checkpoint_dir=checkpoint_dir,
    )
