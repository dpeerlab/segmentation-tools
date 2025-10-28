from skimage.metrics import structural_similarity as ssim
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from skimage.measure import label, regionprops
import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger


def find_poorly_aligned_regions(
    fixed_img,
    moving_img,
    ssim_bounds=(0.0, 0.6),
    win_size=11,
    min_brightness_factor=0.15,
    min_area_factor=5e-5,
    output_file_path=None,
):

    _, ssim_full = ssim(
        fixed_img,
        moving_img,
        data_range=fixed_img.max() - moving_img.min(),
        full=True,
        win_size=win_size,
    )
    min_brightness = min(fixed_img.max(), moving_img.max()) * min_brightness_factor

    masked = np.where(
        (ssim_full >= ssim_bounds[0]) & (ssim_full <= ssim_bounds[1]), 1, 0
    ).astype(np.uint8)

    # Only keep values where both warped_if_ds and xenium_dapi_ds >= 50
    condition = (fixed_img >= min_brightness) | (moving_img >= min_brightness)

    # Set masked = 1 only where it was already 1 and condition is met
    masked_conditioned = masked & condition

    # Generate labeled regions
    labeled_mask = label(masked_conditioned, connectivity=2)
    regions = regionprops(labeled_mask)

    min_area_threshold = (
        min_area_factor * masked_conditioned.shape[0] * masked_conditioned.shape[1]
    )
    filtered_regions = [r for r in regions if r.area >= min_area_threshold]

    # Step 1: Convert bounding boxes to shapely rectangles
    bounding_boxes = []
    for r in filtered_regions:
        minr, minc, maxr, maxc = r.bbox
        bounding_boxes.append(box(minc, minr, maxc, maxr))  # note x/y reversal

    # Step 2: Merge overlapping or touching boxes using shapely
    buffer_distance = min_area_threshold / 5

    # Expand each box by 100 pixels
    expanded_boxes = [b.buffer(buffer_distance) for b in bounding_boxes]

    # Merge overlapping/touching buffered boxes
    merged = unary_union(expanded_boxes)

    # Optional: shrink boxes back to original size (i.e., remove the buffer)
    if merged.geom_type == "Polygon":
        merged_boxes = [merged.buffer(-buffer_distance)]
    else:
        merged_boxes = [g.buffer(-buffer_distance) for g in merged.geoms]

    # Step 4: Plot merged boxes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(masked_conditioned, cmap="gray")

    for poly in merged_boxes:
        minx, miny, maxx, maxy = poly.bounds
        rect = mpl.patches.Rectangle(
            (minx, miny),
            maxx - minx,
            maxy - miny,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        ax.add_patch(rect)

    ax.set_title("Merged Bounding Boxes")
    plt.axis("off")

    if output_file_path:
        fig.savefig(output_file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Poorly aligned regions saved to: {output_file_path}")
    return merged_boxes
