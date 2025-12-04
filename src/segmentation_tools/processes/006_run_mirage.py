import sys
from loguru import logger
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
import mirage
import argparse
import os

def run_mirage(
    warped_image,
    fixed_image,
    bin_mask=None,
    pad=12,
    offset=12,
    num_neurons=200,
    num_layers=3,
    pool=1,
    loss="SSIM",
    batch_size=1024,
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
        batch_size=batch_size,
        LR=lr,
    )

    logger.info("Training MIRAGE model...")
    mirage_model.train(num_steps=num_steps)

    logger.info("MIRAGE model training complete. Computing transform...")

    mirage_model.compute_transform()

    return mirage_model.get_transform()

def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Run MIRAGE alignment on warped and fixed images."
    )

    parser.add_argument(
        "--warped-file-path",
        required=True,
        type=str,
        help="Path to the warped image .npy file.",
    )
    parser.add_argument(
        "--fixed-file-path",
        required=True,
        type=str,
        help="Path to the fixed image .npy file.",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=256,
        help="Batch size for MIRAGE training.",
    )
    parser.add_argument(
        "--learning-rate",
        required=False,
        type=float,
        default=0.001,
        help="Learning rate for MIRAGE training.",
    )

    return parser.parse_args()


def main(warped_file_path, fixed_file_path, checkpoint_dir):
    warped_image = np.load(warped_file_path)
    mirage_transform_file_path = checkpoint_dir / "mirage_transform.npy"
    if os.path.exists(mirage_transform_file_path):
        logger.info(f"MIRAGE transform already exists at {mirage_transform_file_path}. Skipping computation.")
        return 0
        
    mirage_transform = run_mirage(
        warped_image=warped_image,
        fixed_image=np.load(fixed_file_path),
        pad=13,
        offset=15,
        pool=1,
        loss="SSIM",
    )

    if mirage_transform is None:
        logger.error("MIRAGE alignment failed.")
        return 1
    np.save(mirage_transform_file_path, mirage_transform)
    logger.info(f"MIRAGE transform saved to {mirage_transform_file_path}")
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    warped_file_path = args.warped_file_path
    fixed_file_path = args.fixed_file_path
    batch_size = args.batch_size
    lr = args.learning_rate

    checkpoint_dir = Path(warped_file_path).parent
    main(
        warped_file_path=warped_file_path,
        fixed_file_path=fixed_file_path,
        checkpoint_dir=checkpoint_dir,
    )
