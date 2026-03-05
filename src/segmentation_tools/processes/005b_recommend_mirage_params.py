"""Pre-MIRAGE analysis: sample tissue crops, recommend hyperparameters, and
visualize centroid displacement fields.

This step runs *after* the SIFT warp (step 005) and *before* MIRAGE (step 006).
It reads the linearly-warped moving image and the high-res fixed image, samples
1000x1000 crops that contain tissue, and for each crop:

1. Segments nuclei via connected-component analysis on the thresholded image.
2. Matches centroids between fixed and moving via KDTree nearest-neighbour.
3. Rejects outlier displacements using MAD (Median Absolute Deviation).
4. Recommends MIRAGE hyperparameters (offset, pad, smoothness_radius,
   pos_encoding_L, dissim_sigma) from the displacement statistics and
   nucleus size.

Outputs
-------
- ``recommended_mirage_params.npy`` : dict saved via ``np.save(..., allow_pickle=True)``
- ``mirage_param_recommendation.png`` : 6-panel summary figure
- ``centroid_quiver_crop_*.png`` : per-crop quiver plots of displacement fields
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# Crop sampling
# ---------------------------------------------------------------------------

def sample_tissue_crops(image, crop_size=1000, n_crops=5, min_tissue_fraction=0.15,
                        max_attempts_factor=10, rng=None):
    """Sample random crops that contain enough tissue (non-zero signal).

    Parameters
    ----------
    image : np.ndarray (2D)
    crop_size : int
    n_crops : int
    min_tissue_fraction : float
        Minimum fraction of pixels with signal (> 0) to accept a crop.
    max_attempts_factor : int
        ``max_attempts = n_crops * max_attempts_factor``.
    rng : np.random.Generator or None

    Returns
    -------
    list of dict
        Each dict has keys ``'row'``, ``'col'`` (top-left), ``'crop'``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    H, W = image.shape
    if H < crop_size or W < crop_size:
        logger.warning(
            f"Image ({H}x{W}) smaller than crop_size ({crop_size}). "
            "Using full image as a single crop."
        )
        return [{"row": 0, "col": 0, "crop": image}]

    crops = []
    max_attempts = n_crops * max_attempts_factor
    attempts = 0

    while len(crops) < n_crops and attempts < max_attempts:
        attempts += 1
        r = rng.integers(0, H - crop_size + 1)
        c = rng.integers(0, W - crop_size + 1)
        crop = image[r : r + crop_size, c : c + crop_size]

        tissue_frac = np.count_nonzero(crop) / crop.size
        if tissue_frac >= min_tissue_fraction and np.var(crop) > 1e-4:
            crops.append({"row": r, "col": c, "crop": crop})

    if not crops:
        logger.warning("No crops passed tissue filter. Taking best-effort crop from image centre.")
        r = max(0, H // 2 - crop_size // 2)
        c = max(0, W // 2 - crop_size // 2)
        crops.append({"row": r, "col": c, "crop": image[r : r + crop_size, c : c + crop_size]})

    logger.info(f"Sampled {len(crops)} tissue crops out of {attempts} attempts.")
    return crops


# ---------------------------------------------------------------------------
# Centroid extraction & matching
# ---------------------------------------------------------------------------

def get_centroids(img, min_area=20):
    """Return (N, 2) array of (row, col) centroids from a thresholded image."""
    mask = (img > 0).astype(np.uint8)
    labeled = label(mask)
    props = regionprops(labeled)
    centroids = [p.centroid for p in props if p.area >= min_area]
    return np.array(centroids) if centroids else np.empty((0, 2))


def get_centroid_areas(img, min_area=20):
    """Return centroids and their areas."""
    mask = (img > 0).astype(np.uint8)
    labeled = label(mask)
    props = regionprops(labeled)
    centroids = []
    areas = []
    for p in props:
        if p.area >= min_area:
            centroids.append(p.centroid)
            areas.append(p.area)
    return (
        np.array(centroids) if centroids else np.empty((0, 2)),
        np.array(areas) if areas else np.empty((0,)),
    )


def match_centroids(src, dst, max_dist=100):
    """Nearest-neighbour matching of *src* centroids to *dst*.

    Returns
    -------
    distances, src_matched, dst_matched : np.ndarray
    """
    if len(src) == 0 or len(dst) == 0:
        return np.array([]), np.empty((0, 2)), np.empty((0, 2))
    tree = cKDTree(dst)
    dists, indices = tree.query(src)
    valid = dists < max_dist
    return dists[valid], src[valid], dst[indices[valid]]


# ---------------------------------------------------------------------------
# Parameter recommendation (per-crop)
# ---------------------------------------------------------------------------

def recommend_mirage_params(
    fixed_crop,
    moving_crop,
    min_nucleus_area=50,
    max_match_distance=100,
    mad_outlier_factor=3.0,
):
    """Analyze a single crop pair and recommend MIRAGE hyperparameters.

    Returns dict with recommended params and matched-centroid data.
    """
    fixed_centroids, fixed_areas = get_centroid_areas(fixed_crop, min_area=min_nucleus_area)
    moving_centroids, moving_areas = get_centroid_areas(moving_crop, min_area=min_nucleus_area)

    n_fixed = len(fixed_centroids)
    n_moving = len(moving_centroids)

    defaults = {
        "offset": 30, "pad": 30, "smoothness_radius": 50,
        "pos_encoding_L": 6, "dissim_sigma": 30,
        "median_displacement": np.array([0.0, 0.0]),
        "displacement_magnitude": 0.0,
        "num_matched": 0, "num_fixed_nuclei": n_fixed, "num_moving_nuclei": n_moving,
        "matched_centroids": {"fixed": np.empty((0, 2)), "moving": np.empty((0, 2)),
                              "displacements": np.empty((0, 2))},
    }

    if n_fixed < 3 or n_moving < 3:
        logger.warning(f"Too few nuclei (fixed={n_fixed}, moving={n_moving}). Using defaults.")
        return defaults

    # Nearest-neighbour matching (moving -> fixed)
    fixed_tree = cKDTree(fixed_centroids)
    distances, indices = fixed_tree.query(moving_centroids, k=1)
    valid = distances < max_match_distance

    matched_moving = moving_centroids[valid]
    matched_fixed = fixed_centroids[indices[valid]]
    raw_displacements = matched_fixed - matched_moving

    if len(raw_displacements) < 3:
        logger.warning(f"Too few matches ({len(raw_displacements)}). Using defaults.")
        return defaults

    # Outlier rejection via MAD
    median_disp = np.median(raw_displacements, axis=0)
    deviations = np.linalg.norm(raw_displacements - median_disp, axis=1)
    mad = median_abs_deviation(deviations)

    if mad < 1e-6:
        inlier_mask = np.ones(len(deviations), dtype=bool)
    else:
        median_dev = np.median(deviations)
        inlier_mask = deviations < (median_dev + mad_outlier_factor * mad)

    inlier_displacements = raw_displacements[inlier_mask]
    inlier_moving = matched_moving[inlier_mask]
    inlier_fixed = matched_fixed[inlier_mask]

    median_disp = np.median(inlier_displacements, axis=0)
    magnitudes = np.linalg.norm(inlier_displacements, axis=1)
    median_mag = float(np.median(magnitudes))
    p90_mag = float(np.percentile(magnitudes, 90))
    disp_std = float(np.std(magnitudes))

    # Nucleus diameter
    all_areas = np.concatenate([fixed_areas, moving_areas])
    avg_nucleus_diameter = float(np.sqrt(np.median(all_areas) / np.pi) * 2)

    # Recommend parameters
    recommended_offset = max(int(np.ceil(p90_mag * 1.3)), 10)
    recommended_pad = max(recommended_offset, int(np.ceil(avg_nucleus_diameter * 0.8)))
    recommended_smoothness_radius = max(int(np.ceil(avg_nucleus_diameter * 2.5)), 30)

    if disp_std < 3:
        recommended_L = 4
    elif disp_std < 10:
        recommended_L = 6
    else:
        recommended_L = 8

    recommended_dissim_sigma = max(int(np.ceil(avg_nucleus_diameter * 0.5)), recommended_offset)

    return {
        "offset": recommended_offset,
        "pad": recommended_pad,
        "smoothness_radius": recommended_smoothness_radius,
        "pos_encoding_L": recommended_L,
        "dissim_sigma": recommended_dissim_sigma,
        "median_displacement": median_disp,
        "displacement_magnitude": median_mag,
        "p90_magnitude": p90_mag,
        "displacement_std": disp_std,
        "avg_nucleus_diameter": avg_nucleus_diameter,
        "num_matched": int(inlier_mask.sum()),
        "num_fixed_nuclei": n_fixed,
        "num_moving_nuclei": n_moving,
        "matched_centroids": {
            "fixed": inlier_fixed,
            "moving": inlier_moving,
            "displacements": inlier_displacements,
        },
    }


# ---------------------------------------------------------------------------
# Aggregation across crops
# ---------------------------------------------------------------------------

def aggregate_recommendations(crop_results):
    """Take per-crop recommendation dicts and produce a single consensus."""
    keys = ["offset", "pad", "smoothness_radius", "pos_encoding_L", "dissim_sigma"]
    agg = {}
    for k in keys:
        values = [r[k] for r in crop_results if r["num_matched"] > 0]
        if values:
            agg[k] = int(np.median(values))
        else:
            agg[k] = crop_results[0][k]  # fallback to first default

    # Aggregate stats
    mag_values = [r["displacement_magnitude"] for r in crop_results if r["num_matched"] > 0]
    agg["median_displacement_magnitude"] = float(np.median(mag_values)) if mag_values else 0.0
    agg["num_crops_with_matches"] = sum(1 for r in crop_results if r["num_matched"] > 0)
    agg["total_inlier_matches"] = sum(r["num_matched"] for r in crop_results)

    return agg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_quiver(fixed_crop, moving_crop, params, crop_idx, save_path):
    """Draw centroid displacement quiver plot for a single crop."""
    matched = params["matched_centroids"]
    if len(matched["fixed"]) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Overlay
    from segmentation_tools.utils.utils import create_rgb_overlay
    overlay = create_rgb_overlay(fixed_crop, moving_crop)
    axes[0].imshow(overlay)

    fc = matched["fixed"]
    mc = matched["moving"]
    for i in range(len(fc)):
        axes[0].plot([mc[i, 1], fc[i, 1]], [mc[i, 0], fc[i, 0]], "w-", lw=0.5, alpha=0.6)
    axes[0].scatter(mc[:, 1], mc[:, 0], c="red", s=8, zorder=5, label="Moving")
    axes[0].scatter(fc[:, 1], fc[:, 0], c="cyan", s=8, zorder=5, label="Fixed")
    axes[0].set_title(f"Crop {crop_idx} — Centroid Matches\n"
                      f"mean dist: {params['displacement_magnitude']:.1f}px")
    axes[0].legend(fontsize=8)
    axes[0].axis("off")

    # Quiver
    disps = matched["displacements"]
    axes[1].imshow(moving_crop, cmap="gray", alpha=0.3)
    mag = np.linalg.norm(disps, axis=1)
    q = axes[1].quiver(
        mc[:, 1], mc[:, 0],
        disps[:, 1], disps[:, 0],
        mag, cmap="jet", angles="xy", scale_units="xy", scale=1,
        width=0.003, headwidth=3, headlength=4,
    )
    plt.colorbar(q, ax=axes[1], label="Displacement (px)")
    axes[1].set_xlim(0, moving_crop.shape[1])
    axes[1].set_ylim(moving_crop.shape[0], 0)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Displacement Quiver\nmedian={params['displacement_magnitude']:.1f}px, "
                      f"p90={params.get('p90_magnitude', 0):.1f}px")

    # Histogram
    dists = np.linalg.norm(disps, axis=1)
    axes[2].hist(dists, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    axes[2].axvline(np.median(dists), color="orange", ls="--", lw=2,
                    label=f"Median: {np.median(dists):.1f}px")
    axes[2].axvline(params["offset"], color="green", ls="-", lw=2.5,
                    label=f"offset={params['offset']}")
    axes[2].set_xlabel("Displacement (px)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Displacement Distribution")
    axes[2].legend(fontsize=8)

    plt.suptitle(f"Pre-MIRAGE Crop {crop_idx} Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved quiver plot to {save_path}")


def plot_summary(agg_params, crop_results, save_path):
    """Plot a summary figure with the aggregated parameter table."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axis("off")

    table_data = [
        ["Parameter", "Value", "Rationale"],
        ["offset", str(agg_params["offset"]), "1.3 x P90 displacement (median across crops)"],
        ["pad", str(agg_params["pad"]), "max(offset, 0.8 x nucleus diameter)"],
        ["smoothness_radius", str(agg_params["smoothness_radius"]), "2.5 x nucleus diameter"],
        ["pos_encoding_L", str(agg_params["pos_encoding_L"]), "Based on displacement variability"],
        ["dissim_sigma", str(agg_params["dissim_sigma"]), "max(0.5 x nucleus diameter, offset)"],
        ["", "", ""],
        ["Metric", "Value", ""],
        ["Crops with matches", str(agg_params["num_crops_with_matches"]), ""],
        ["Total inlier matches", str(agg_params["total_inlier_matches"]), ""],
        ["Median displacement", f"{agg_params['median_displacement_magnitude']:.1f}px", ""],
    ]

    table = ax.table(cellText=table_data, cellLoc="left", loc="center",
                     colWidths=[0.35, 0.2, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for col in range(3):
        table[0, col].set_facecolor("#4472C4")
        table[0, col].set_text_props(color="white", fontweight="bold")
        table[7, col].set_facecolor("#4472C4")
        table[7, col].set_text_props(color="white", fontweight="bold")
        table[6, col].set_facecolor("white")
        table[6, col].set_edgecolor("white")

    plt.title("Recommended MIRAGE Parameters (aggregated across crops)",
              fontsize=13, fontweight="bold", pad=20)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved parameter summary to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(warped_file_path, fixed_file_path, checkpoint_dir, results_dir=None,
         crop_size=1000, n_crops=5):
    """Sample crops, recommend MIRAGE params, save results and plots."""
    checkpoint_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir) if results_dir else checkpoint_dir

    warped_image = np.load(warped_file_path)
    fixed_image = np.load(fixed_file_path)

    logger.info(f"Warped image shape: {warped_image.shape}")
    logger.info(f"Fixed image shape:  {fixed_image.shape}")

    # Use the smaller of the two shapes for crop bounds
    min_h = min(warped_image.shape[0], fixed_image.shape[0])
    min_w = min(warped_image.shape[1], fixed_image.shape[1])
    warped_image = warped_image[:min_h, :min_w]
    fixed_image = fixed_image[:min_h, :min_w]

    # Sample crops from fixed image (tissue presence check)
    rng = np.random.default_rng(42)
    fixed_crops = sample_tissue_crops(fixed_image, crop_size=crop_size,
                                      n_crops=n_crops, rng=rng)

    crop_results = []
    for i, crop_info in enumerate(fixed_crops):
        r, c = crop_info["row"], crop_info["col"]
        fixed_crop = crop_info["crop"]
        moving_crop = warped_image[r : r + crop_size, c : c + crop_size]

        logger.info(f"Crop {i}: row={r}, col={c}, size={fixed_crop.shape}")

        params = recommend_mirage_params(fixed_crop, moving_crop)
        crop_results.append(params)

        logger.info(
            f"  Crop {i}: matched={params['num_matched']}, "
            f"median_disp={params['displacement_magnitude']:.1f}px, "
            f"offset={params['offset']}, pad={params['pad']}"
        )

        # Quiver plot per crop
        plot_quiver(
            fixed_crop, moving_crop, params, crop_idx=i,
            save_path=results_dir / f"centroid_quiver_crop_{i}.png",
        )

    # Aggregate
    agg = aggregate_recommendations(crop_results)
    logger.info(f"Aggregated recommendation: {agg}")

    # Save
    out_path = checkpoint_dir / "recommended_mirage_params.npy"
    np.save(out_path, agg, allow_pickle=True)
    logger.info(f"Saved recommended params to {out_path}")

    plot_summary(agg, crop_results, results_dir / "mirage_param_recommendation.png")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Recommend MIRAGE hyperparameters from tissue crops."
    )
    parser.add_argument("--warped-file-path", required=True, type=str,
                        help="Path to linearly-warped moving DAPI .npy file.")
    parser.add_argument("--fixed-file-path", required=True, type=str,
                        help="Path to high-res fixed DAPI .npy file.")
    parser.add_argument("--crop-size", type=int, default=1000,
                        help="Side length of square crops (default: 1000).")
    parser.add_argument("--n-crops", type=int, default=5,
                        help="Number of tissue crops to sample (default: 5).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    checkpoint_dir = Path(args.warped_file_path).parent
    main(
        warped_file_path=args.warped_file_path,
        fixed_file_path=args.fixed_file_path,
        checkpoint_dir=checkpoint_dir,
        crop_size=args.crop_size,
        n_crops=args.n_crops,
    )
