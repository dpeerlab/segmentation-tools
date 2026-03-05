from loguru import logger
import numpy as np
from pathlib import Path
import mirage
import argparse


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
    smoothness_weight=0.1,
    smoothness_radius=30,
    pos_encoding_L=6,
    dissim_sigma=None,
):
    """Run MIRAGE alignment on the given images.

    Returns the MIRAGE transform as a numpy array.
    """
    logger.info("Running MIRAGE alignment...")
    logger.info(f"  pad={pad}, offset={offset}, pool={pool}, "
                f"smoothness_radius={smoothness_radius}, "
                f"pos_encoding_L={pos_encoding_L}, dissim_sigma={dissim_sigma}")

    mirage_model = mirage.MIRAGE(
        images=warped_image,
        references=fixed_image,
        bin_mask=bin_mask,
        pad=pad,
        offset=offset,
        num_neurons=num_neurons,
        num_layers=num_layers,
        pool=pool,
        loss_type=loss,
        batch_size=batch_size,
        LR=lr,
        smoothness_weight=smoothness_weight,
        smoothness_radius=smoothness_radius,
        pos_encoding_L=pos_encoding_L,
        dissim_sigma=dissim_sigma,
    )

    logger.info("Training MIRAGE model...")
    mirage_model.train(num_steps=num_steps)

    logger.info("MIRAGE model training complete. Computing transform...")

    mirage_model.compute_transform()

    return mirage_model.get_transform()


def _load_recommended_params(checkpoint_dir):
    """Load recommended params from 005b if available, else return defaults."""
    params_path = Path(checkpoint_dir) / "recommended_mirage_params.npy"
    defaults = {
        "offset": 15,
        "pad": 13,
        "smoothness_radius": 30,
        "pos_encoding_L": 6,
        "dissim_sigma": None,
    }
    if params_path.exists():
        recommended = np.load(params_path, allow_pickle=True).item()
        logger.info(f"Loaded recommended MIRAGE params from {params_path}")
        for key in defaults:
            if key in recommended:
                defaults[key] = recommended[key]
        logger.info(f"  Using: {defaults}")
    else:
        logger.warning(f"No recommended params found at {params_path}. Using defaults.")
    return defaults

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
        default=0.012575,
        help="Learning rate for MIRAGE training.",
    )
    parser.add_argument(
        "--num-steps",
        required=False,
        type=int,
        default=2048,
        help="Number of MIRAGE training steps.",
    )

    return parser.parse_args()


def main(warped_file_path, fixed_file_path, checkpoint_dir,
         batch_size=1024, lr=0.012575, num_steps=2048):
    checkpoint_dir = Path(checkpoint_dir)
    warped_image = np.load(warped_file_path)
    mirage_transform_file_path = checkpoint_dir / "mirage_transform.npy"
    if mirage_transform_file_path.exists():
        logger.info(f"MIRAGE transform already exists at {mirage_transform_file_path}. Skipping.")
        return 0

    recommended = _load_recommended_params(checkpoint_dir)

    mirage_transform = run_mirage(
        warped_image=warped_image,
        fixed_image=np.load(fixed_file_path),
        pad=recommended["pad"],
        offset=recommended["offset"],
        pool=1,
        loss="SSIM",
        batch_size=batch_size,
        num_steps=num_steps,
        lr=lr,
        smoothness_radius=recommended["smoothness_radius"],
        pos_encoding_L=recommended["pos_encoding_L"],
        dissim_sigma=recommended["dissim_sigma"],
    )

    np.save(mirage_transform_file_path, mirage_transform)
    logger.info(f"MIRAGE transform saved to {mirage_transform_file_path}")
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    checkpoint_dir = Path(args.warped_file_path).parent
    main(
        warped_file_path=args.warped_file_path,
        fixed_file_path=args.fixed_file_path,
        checkpoint_dir=checkpoint_dir,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        num_steps=args.num_steps,
    )
