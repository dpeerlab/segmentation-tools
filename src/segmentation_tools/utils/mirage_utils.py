import os
import mirage
import tifffile
from segmentation_tools.utils.image_utils import match_image_histograms

from segmentation_tools.logger import logger


def run_mirage(
    moving_img,
    fixed_img,
    bin_mask=None,
    pad=12,
    offset=12,
    num_neurons=300,
    num_layers=3,
    pool=1,
    loss="SSIM",
    save_img_dir=None,
):
    """
    Run MIRAGE alignment on the given images.

    Parameters:
        moving_img: The moving image to align.
        fixed_img: The fixed image to align against.
        bin_mask: Optional binary mask for the moving image.
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
        images=moving_img,
        references=fixed_img,
        bin_mask=bin_mask,
        pad=pad,
        offset=offset,
        num_neurons=num_neurons,
        num_layers=num_layers,
        pool=pool,
        loss=loss,
    )

    logger.info("Training MIRAGE model...")
    mirage_model.train(batch_size=256, num_steps=512, lr__sched=True, LR=0.005)

    logger.info("MIRAGE model training complete. Computing transform...")
    mirage_model.compute_transform()

    logger.info("Applying transform to moving image...")
    aligned_image = mirage_model.apply_transform(moving_img)
    logger.info("MIRAGE image alignment complete.")

    if save_img_dir:
        output_path = os.path.join(save_img_dir, "mirage_aligned_image.tiff")
        tifffile.imwrite(output_path, aligned_image)

    return aligned_image
