"""Generate an HTML QC report for a completed pipeline run.

Usage:
    segmentation-tools view -o /path/to/output_root -j my_sample
    # Opens results/qc_report.html in browser, or prints path if headless
"""

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_b64(arr: np.ndarray, cmap="gray", vmin=None, vmax=None) -> str:
    """Convert a 2D/3D numpy array to a base64-encoded PNG for embedding in HTML."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    if arr.ndim == 2:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    else:
        ax.imshow(arr, interpolation="nearest")
    plt.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=80)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _png_to_b64(path: Path) -> str:
    """Read an existing PNG file and return base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _thumbnail(arr: np.ndarray, max_px: int = 512) -> np.ndarray:
    """Downsample a 2D array to at most max_px on the longest side."""
    h, w = arr.shape[:2]
    scale = max_px / max(h, w)
    if scale >= 1:
        return arr
    new_h, new_w = int(h * scale), int(w * scale)
    from skimage.transform import resize
    return resize(arr, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(arr.dtype)


def _overlay_rgb(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Magenta/green overlay: fixed=green, moving=magenta."""
    f = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
    m = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
    rgb = np.stack([m, f, m], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_metrics(checkpoint_dir: Path, results_dir: Path) -> dict:
    """Gather all available QC metrics from checkpoint files."""
    metrics = {}

    # Pipeline status
    status_file = checkpoint_dir / "pipeline_status.txt"
    if status_file.exists():
        parts = status_file.read_text().strip().split("|")
        if len(parts) >= 4:
            metrics["last_step"] = parts[0]
            metrics["last_step_name"] = parts[1]
            metrics["last_step_time"] = parts[2]
            metrics["pipeline_status"] = parts[3]

    # Linear transform
    lt_path = checkpoint_dir / "linear_transform.npy"
    if lt_path.exists():
        M = np.load(lt_path)
        metrics["linear_transform"] = M.tolist()
        # Extract rotation angle and scale from 2x2 upper-left
        a, b = M[0, 0], M[0, 1]
        angle_deg = float(np.degrees(np.arctan2(b, a)))
        scale = float(np.sqrt(a**2 + b**2))
        metrics["transform_rotation_deg"] = round(angle_deg, 1)
        metrics["transform_scale"] = round(scale, 4)
        metrics["transform_translation_xy"] = [round(M[0, 2], 1), round(M[1, 2], 1)]

    # MIRAGE params used
    mp_path = checkpoint_dir / "recommended_mirage_params.npy"
    if mp_path.exists():
        mp = np.load(mp_path, allow_pickle=True).item()
        metrics["mirage_params"] = {k: v for k, v in mp.items()
                                     if not isinstance(v, np.ndarray)}

    # Cell counts from mask files
    for prefix in ["membrane_dapi", "dapi", "membrane"]:
        mask_path = results_dir / f"{prefix}_segmentation_masks.npy"
        if mask_path.exists():
            masks = np.load(mask_path)
            metrics[f"cell_count_{prefix}"] = int(masks.max())

    return metrics


def _collect_images(checkpoint_dir: Path, results_dir: Path) -> dict:
    """Collect image thumbnails available at report time."""
    images = {}

    # VALIS overlap images
    valis_overlap_dir = checkpoint_dir / "valis_output" / "valis_input" / "overlaps"
    for name, key in [
        ("valis_input_original_overlap.png", "valis_before"),
        ("valis_input_rigid_overlap.png", "valis_after"),
    ]:
        p = valis_overlap_dir / name
        if p.exists():
            images[key] = _png_to_b64(p)

    # Pre/post MIRAGE warped DAPI overlay
    fixed_path = checkpoint_dir / "high_res_fixed_dapi_filtered_level_0.npy"
    warped_path = checkpoint_dir / "moving_dapi_linear_warped.npy"
    if fixed_path.exists() and warped_path.exists():
        fixed = _thumbnail(np.load(fixed_path))
        warped = _thumbnail(np.load(warped_path))
        # Crop to common shape
        h = min(fixed.shape[0], warped.shape[0])
        w = min(fixed.shape[1], warped.shape[1])
        images["pre_mirage_overlay"] = _img_to_b64(_overlay_rgb(fixed[:h, :w], warped[:h, :w]))

    # MIRAGE evaluation quiver plots (crop 0)
    quiver_path = results_dir / "centroid_quiver_crop_0.png"
    if quiver_path.exists():
        images["quiver_crop_0"] = _png_to_b64(quiver_path)

    eval_path = results_dir / "centroid_eval_crop_0.png"
    if eval_path.exists():
        images["eval_crop_0"] = _png_to_b64(eval_path)

    eval_summary = results_dir / "centroid_eval_summary.png"
    if eval_summary.exists():
        images["eval_summary"] = _png_to_b64(eval_summary)

    mirage_param_summary = results_dir / "mirage_param_recommendation.png"
    if mirage_param_summary.exists():
        images["mirage_param_summary"] = _png_to_b64(mirage_param_summary)

    return images


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>segmentation-tools QC — {job_title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1   {{ color: #fff; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  h2   {{ color: #aaa; font-size: 1rem; text-transform: uppercase;
         letter-spacing: 0.08em; margin-top: 2rem; }}
  .card  {{ background: #1a1a1a; border-radius: 8px; padding: 16px;
            margin-bottom: 16px; border: 1px solid #2a2a2a; }}
  .grid  {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 16px; }}
  .grid img {{ width: 100%; border-radius: 4px; }}
  .pill  {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.8rem; font-weight: 600; }}
  .ok    {{ background: #1a3a1a; color: #6fcf97; }}
  .warn  {{ background: #3a2a1a; color: #f2c94c; }}
  .run   {{ background: #1a2a3a; color: #56ccf2; }}
  table  {{ border-collapse: collapse; width: 100%; }}
  td, th {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid #2a2a2a; }}
  th     {{ color: #888; font-weight: normal; font-size: 0.85rem; }}
  .mono  {{ font-family: monospace; font-size: 0.9rem; }}
  .caption {{ color: #666; font-size: 0.8rem; margin-top: 4px; }}
</style>
</head>
<body>
<h1>QC Report — {job_title}</h1>
<p style="color:#666">Generated {timestamp} &nbsp;|&nbsp; Output: <span class="mono">{output_path}</span></p>

{status_section}
{metrics_section}
{images_section}

</body>
</html>"""


def _render_status(metrics: dict) -> str:
    status = metrics.get("pipeline_status", "unknown")
    pill_cls = {"done": "ok", "running": "run", "complete": "ok"}.get(status, "warn")
    last = metrics.get("last_step_name", "?")
    t = metrics.get("last_step_time", "")
    return f"""
<div class="card">
  <h2>Pipeline Status</h2>
  <span class="pill {pill_cls}">{status.upper()}</span>
  &nbsp; Last step: <strong>{last}</strong>
  <span style="color:#555; margin-left:12px">{t}</span>
</div>"""


def _render_metrics(metrics: dict) -> str:
    rows = []

    if "transform_rotation_deg" in metrics:
        rows.append(("Rotation (deg)", f"{metrics['transform_rotation_deg']}°"))
        rows.append(("Scale factor", metrics["transform_scale"]))
        tx, ty = metrics["transform_translation_xy"]
        rows.append(("Translation (x, y px)", f"{tx}, {ty}"))

    for prefix, label in [("membrane_dapi", "Combined (DAPI+membrane)"),
                           ("dapi", "DAPI-only"), ("membrane", "Membrane-only")]:
        key = f"cell_count_{prefix}"
        if key in metrics:
            rows.append((f"Cells detected — {label}", f"{metrics[key]:,}"))

    if "mirage_params" in metrics:
        mp = metrics["mirage_params"]
        rows.append(("MIRAGE offset", mp.get("offset", "—")))
        rows.append(("MIRAGE smoothness_radius", mp.get("smoothness_radius", "—")))
        rows.append(("MIRAGE median displacement (px)",
                     f"{mp.get('median_displacement_magnitude', 0):.1f}"))

    if not rows:
        return ""

    table_rows = "\n".join(f"<tr><th>{k}</th><td class='mono'>{v}</td></tr>" for k, v in rows)
    return f"""
<div class="card">
  <h2>Key Metrics</h2>
  <table>{table_rows}</table>
</div>"""


def _render_images(images: dict) -> str:
    if not images:
        return ""

    sections = []

    # Alignment overview
    align_imgs = []
    for key, caption in [
        ("valis_before", "Before VALIS alignment"),
        ("valis_after", "After VALIS alignment"),
        ("pre_mirage_overlay", "After VALIS — pre-MIRAGE overlay (green=fixed, magenta=moving)"),
    ]:
        if key in images:
            align_imgs.append(
                f"<div><img src='data:image/png;base64,{images[key]}'>"
                f"<div class='caption'>{caption}</div></div>"
            )
    if align_imgs:
        sections.append(
            "<h2>Alignment Quality</h2>"
            f"<div class='grid'>{''.join(align_imgs)}</div>"
        )

    # MIRAGE evaluation
    mirage_imgs = []
    for key, caption in [
        ("mirage_param_summary", "Recommended MIRAGE parameters"),
        ("quiver_crop_0", "Pre-MIRAGE displacement quiver (crop 0)"),
        ("eval_crop_0", "Before/after MIRAGE centroid alignment (crop 0)"),
        ("eval_summary", "Mean centroid distance before vs after MIRAGE"),
    ]:
        if key in images:
            mirage_imgs.append(
                f"<div><img src='data:image/png;base64,{images[key]}'>"
                f"<div class='caption'>{caption}</div></div>"
            )
    if mirage_imgs:
        sections.append(
            "<h2>MIRAGE Non-linear Correction</h2>"
            f"<div class='grid'>{''.join(mirage_imgs)}</div>"
        )

    return "<div class='card'>" + "\n".join(sections) + "</div>"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(output_root: Path, job_title: str) -> Path:
    """Generate QC HTML report. Returns path to the written file."""
    checkpoint_dir = output_root / job_title / ".checkpoints"
    results_dir = output_root / job_title / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting metrics from {checkpoint_dir}")
    metrics = _collect_metrics(checkpoint_dir, results_dir)
    images = _collect_images(checkpoint_dir, results_dir)

    html = _HTML_TEMPLATE.format(
        job_title=job_title,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        output_path=output_root / job_title,
        status_section=_render_status(metrics),
        metrics_section=_render_metrics(metrics),
        images_section=_render_images(images),
    )

    report_path = results_dir / "qc_report.html"
    report_path.write_text(html)
    logger.success(f"QC report written to {report_path}")
    return report_path
