"""Post-MIRAGE evaluation: centroid matching on crops before/after alignment.

This step runs *after* MIRAGE (step 006). It applies the MIRAGE transform to
the same tissue crops used for parameter recommendation, then compares
centroid alignment before and after MIRAGE.

Outputs
-------
- ``centroid_eval_crop_*.png`` : per-crop before/after overlay + distance histogram
- ``centroid_eval_summary.png`` : aggregate distance comparison
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage.transform
from loguru import logger

import importlib
recommend_mod = importlib.import_module(
    "segmentation_tools.processes.005b_recommend_mirage_params"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_transform_to_crop(moving_crop, full_transform, row_offset, col_offset, crop_size):
    """Extract the transform region for a crop and warp it.

    Parameters
    ----------
    moving_crop : np.ndarray (crop_size, crop_size)
    full_transform : np.ndarray (H, W, 2)
        MIRAGE transform where [:,:,0] = row coords, [:,:,1] = col coords.
    row_offset, col_offset : int
        Top-left corner of the crop in the full image.
    crop_size : int

    Returns
    -------
    np.ndarray : warped crop
    """
    r0, c0 = row_offset, col_offset
    r1 = min(r0 + crop_size, full_transform.shape[0])
    c1 = min(c0 + crop_size, full_transform.shape[1])

    # Extract sub-transform and shift to crop-local coordinates
    sub_transform = full_transform[r0:r1, c0:c1].copy()
    sub_transform[:, :, 0] -= r0
    sub_transform[:, :, 1] -= c0

    warped = skimage.transform.warp(
        moving_crop.astype("float32"),
        np.array([sub_transform[:, :, 0], sub_transform[:, :, 1]]),
        order=1,
    )
    return warped


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_before_after(fixed_crop, moving_before, moving_after, crop_idx, save_path):
    """3-panel figure: before overlay, after overlay, distance histogram."""
    from segmentation_tools.utils.utils import create_rgb_overlay

    c_fixed = recommend_mod.get_centroids(fixed_crop)
    c_before = recommend_mod.get_centroids(moving_before)
    c_after = recommend_mod.get_centroids(moving_after)

    dists_before, matched_before, matched_fixed_b = recommend_mod.match_centroids(c_before, c_fixed)
    dists_after, matched_after, matched_fixed_a = recommend_mod.match_centroids(c_after, c_fixed)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Before
    overlay_before = create_rgb_overlay(fixed_crop, moving_before)
    axes[0].imshow(overlay_before)
    if len(matched_before) > 0:
        for (r1, c1), (r2, c2) in zip(matched_before, matched_fixed_b):
            axes[0].plot([c1, c2], [r1, r2], "w-", lw=0.5, alpha=0.6)
        axes[0].scatter(matched_before[:, 1], matched_before[:, 0], c="red", s=8, zorder=5, label="Moving")
        axes[0].scatter(matched_fixed_b[:, 1], matched_fixed_b[:, 0], c="cyan", s=8, zorder=5, label="Fixed")
    title_before = f"Before MIRAGE\nmean dist: {dists_before.mean():.1f}px" if len(dists_before) > 0 else "Before"
    axes[0].set_title(title_before)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].axis("off")

    # After
    overlay_after = create_rgb_overlay(fixed_crop, moving_after)
    axes[1].imshow(overlay_after)
    if len(matched_after) > 0:
        for (r1, c1), (r2, c2) in zip(matched_after, matched_fixed_a):
            axes[1].plot([c1, c2], [r1, r2], "w-", lw=0.5, alpha=0.6)
        axes[1].scatter(matched_after[:, 1], matched_after[:, 0], c="red", s=8, zorder=5, label="Moving")
        axes[1].scatter(matched_fixed_a[:, 1], matched_fixed_a[:, 0], c="cyan", s=8, zorder=5, label="Fixed")
    title_after = f"After MIRAGE\nmean dist: {dists_after.mean():.1f}px" if len(dists_after) > 0 else "After"
    axes[1].set_title(title_after)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].axis("off")

    # Histogram
    if len(dists_before) > 0 and len(dists_after) > 0:
        bins = np.linspace(0, max(dists_before.max(), dists_after.max()), 30)
        axes[2].hist(dists_before, bins=bins, alpha=0.6,
                     label=f"Before (mu={dists_before.mean():.1f})", color="red")
        axes[2].hist(dists_after, bins=bins, alpha=0.6,
                     label=f"After (mu={dists_after.mean():.1f})", color="green")
        axes[2].axvline(np.median(dists_before), color="red", ls="--", alpha=0.7)
        axes[2].axvline(np.median(dists_after), color="green", ls="--", alpha=0.7)
        axes[2].set_xlabel("Centroid distance (px)")
        axes[2].set_ylabel("Count")
        axes[2].legend(fontsize=9)
        axes[2].set_title("Distance Distribution")

    plt.suptitle(f"Crop {crop_idx} — Centroid Alignment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "crop_idx": crop_idx,
        "n_before": len(dists_before),
        "n_after": len(dists_after),
        "mean_before": float(dists_before.mean()) if len(dists_before) > 0 else np.nan,
        "mean_after": float(dists_after.mean()) if len(dists_after) > 0 else np.nan,
        "median_before": float(np.median(dists_before)) if len(dists_before) > 0 else np.nan,
        "median_after": float(np.median(dists_after)) if len(dists_after) > 0 else np.nan,
    }
    return stats


def plot_aggregate_summary(all_stats, save_path):
    """Bar chart comparing mean centroid distance before/after across crops."""
    valid = [s for s in all_stats if not np.isnan(s["mean_before"]) and not np.isnan(s["mean_after"])]
    if not valid:
        logger.warning("No valid crop stats for summary plot.")
        return

    labels = [f"Crop {s['crop_idx']}" for s in valid]
    before = [s["mean_before"] for s in valid]
    after = [s["mean_after"] for s in valid]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    ax.bar(x - width / 2, before, width, label="Before MIRAGE", color="salmon")
    ax.bar(x + width / 2, after, width, label="After MIRAGE", color="mediumseagreen")
    ax.set_ylabel("Mean centroid distance (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Centroid Alignment: Before vs After MIRAGE")

    # Add improvement text
    for i, s in enumerate(valid):
        improvement = s["mean_before"] - s["mean_after"]
        ax.text(i, max(s["mean_before"], s["mean_after"]) + 0.5,
                f"{improvement:+.1f}px", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved aggregate summary to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(warped_file_path, fixed_file_path, mirage_transform_path,
         checkpoint_dir, results_dir=None, crop_size=1000, n_crops=5):
    """Evaluate MIRAGE alignment quality using centroid matching on tissue crops."""
    checkpoint_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir) if results_dir else checkpoint_dir

    warped_image = np.load(warped_file_path)
    fixed_image = np.load(fixed_file_path)
    mirage_transform = np.load(mirage_transform_path)

    logger.info(f"Warped shape: {warped_image.shape}, Fixed shape: {fixed_image.shape}")
    logger.info(f"MIRAGE transform shape: {mirage_transform.shape}")

    # Crop to common size
    min_h = min(warped_image.shape[0], fixed_image.shape[0])
    min_w = min(warped_image.shape[1], fixed_image.shape[1])
    warped_image = warped_image[:min_h, :min_w]
    fixed_image = fixed_image[:min_h, :min_w]

    # Sample crops (same seed as 005b for reproducibility)
    rng = np.random.default_rng(42)
    fixed_crops = recommend_mod.sample_tissue_crops(
        fixed_image, crop_size=crop_size, n_crops=n_crops, rng=rng
    )

    all_stats = []
    for i, crop_info in enumerate(fixed_crops):
        r, c = crop_info["row"], crop_info["col"]
        fixed_crop = crop_info["crop"]
        moving_before = warped_image[r : r + crop_size, c : c + crop_size]

        # Apply MIRAGE transform to crop
        moving_after = apply_transform_to_crop(
            moving_before, mirage_transform, r, c, crop_size
        )

        stats = plot_before_after(
            fixed_crop, moving_before, moving_after, crop_idx=i,
            save_path=results_dir / f"centroid_eval_crop_{i}.png",
        )
        all_stats.append(stats)

        logger.info(
            f"Crop {i}: mean dist {stats['mean_before']:.1f} -> {stats['mean_after']:.1f}px "
            f"(delta={stats['mean_before'] - stats['mean_after']:+.1f}px)"
        )

    plot_aggregate_summary(all_stats, results_dir / "centroid_eval_summary.png")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate MIRAGE alignment via centroid matching on tissue crops."
    )
    parser.add_argument("--warped-file-path", required=True, type=str,
                        help="Path to linearly-warped moving DAPI .npy file.")
    parser.add_argument("--fixed-file-path", required=True, type=str,
                        help="Path to high-res fixed DAPI .npy file.")
    parser.add_argument("--mirage-transform-path", required=True, type=str,
                        help="Path to MIRAGE transform .npy file.")
    parser.add_argument("--crop-size", type=int, default=1000)
    parser.add_argument("--n-crops", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    checkpoint_dir = Path(args.warped_file_path).parent
    main(
        warped_file_path=args.warped_file_path,
        fixed_file_path=args.fixed_file_path,
        mirage_transform_path=args.mirage_transform_path,
        checkpoint_dir=checkpoint_dir,
        crop_size=args.crop_size,
        n_crops=args.n_crops,
    )
