from pathlib import Path
import os
import tifffile
from skimage.transform import AffineTransform
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shapely_transform
from skimage.transform import ProjectiveTransform

import numpy as np

def get_shape_at_level(
    tiff_path: Path, level: int = 0, series: int = 0
) -> tuple[int, int]:
    """
    Get the shape (height, width) of a given resolution level without loading the image.

    Args:
        tiff_path: Path to the TIFF file.
        level: Resolution level to query (0 = highest).
        series: Series index to use.

    Returns:
        Tuple of (height, width) for the specified level.
    """
    with tifffile.TiffFile(tiff_path) as tf:
        page = tf.series[series].levels[level]
        shape = page.shape

    # If shape includes channel, strip it
    if len(shape) == 3:
        shape = shape[1], shape[2]

    return shape


def get_level_transform(
    path_to_tiff: os.PathLike, level_to: int, level_from: int = 0
) -> AffineTransform:
    """
    Compute the scaling transform (AffineTransform) between two levels of a TIFF pyramid.

    Args:
        path_to_tiff (os.PathLike): Path to the TIFF file.
        level_to (int): Target resolution level.
        level_from (int): Source resolution level (default is 0, i.e. highest res).

    Returns:
        AffineTransform: An affine transform representing the scale between the levels.
    """
    tf = tifffile.TiffFile(path_to_tiff)
    lvl = tf.series[0].levels[level_from]
    from_dims = dict(zip(lvl.axes, lvl.shape))
    lvl = tf.series[0].levels[level_to]
    to_dims = dict(zip(lvl.axes, lvl.shape))
    scale_x = from_dims["X"] / to_dims["X"]
    scale_y = from_dims["Y"] / to_dims["Y"]
    return AffineTransform(scale=(scale_x, scale_y))


def get_crop_transform(
    path_to_tiff: os.PathLike, level_to: int, level_from: int = 0
) -> AffineTransform:
    """
    Compute the affine transform for cropping between two resolution levels of a TIFF file.

    Args:
        path_to_tiff (os.PathLike): Path to the TIFF file.
        level_to (int): Resolution level to crop to.
        level_from (int): Reference resolution level (default is 0).

    Returns:
        AffineTransform: Affine scaling transform between the levels.
    """
    tf = tifffile.TiffFile(path_to_tiff)
    lvl = tf.series[0].levels[level_from]
    from_dims = dict(zip(lvl.axes, lvl.shape))
    lvl = tf.series[0].levels[level_to]
    to_dims = dict(zip(lvl.axes, lvl.shape))
    scale = to_dims["X"] / from_dims["X"], to_dims["Y"] / from_dims["Y"]
    return AffineTransform(scale=scale)


def transform_polygons_to_high_res(polygons, transform):
    """
    Transforms a list of Shapely Polygon or MultiPolygon objects using an image-space transform.

    Args:
        polygons: List of shapely.geometry.Polygon or MultiPolygon
        transform: skimage transform (or raw 3x3 matrix) mapping downsampled to high-res

    Returns:
        List of transformed Polygon or MultiPolygon objects
    """
    if isinstance(transform, np.ndarray):
        transform = ProjectiveTransform(matrix=transform)

    def apply_transform(x, y):
        pts = np.column_stack([x, y])
        transformed = transform(pts)
        return transformed[:, 0], transformed[:, 1]

    transformed_polygons = []
    for poly in polygons:
        if isinstance(poly, (Polygon, MultiPolygon)):
            transformed_poly = shapely_transform(apply_transform, poly)
            transformed_polygons.append(transformed_poly)
        else:
            raise TypeError(f"Expected Polygon or MultiPolygon, got {type(poly)}")

    return transformed_polygons
