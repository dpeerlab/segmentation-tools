import tifffile
from Pathlib import Path
from numpy.typing import ArrayLike
import geopandas as gpd
import numpy as np
import skimage
import shapely
import cv2
import sys

def masks_to_contours(masks: ArrayLike) -> np.ndarray:
    """
    Convert labeled mask image to contours with cell ID annotations.

    Parameters
    ----------
    masks : np.ndarray
        A 2D array of labeled masks where each label corresponds to a cell.

    Returns
    -------
    np.ndarray
        An array of contour points with associated cell IDs.
    """
    # Get contour vertices from masks image
    props = skimage.measure.regionprops(masks.T)
    contours = []
    for p in props:
        # Get largest contour with label
        lbl_contours = cv2.findContours(
            np.pad(p.image, 0).astype('uint8'),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        contour = sorted(lbl_contours, key=lambda c: c.shape[0])[-1]
        if contour.shape[0] > 2:
            contour = np.hstack([
                np.squeeze(contour)[:, ::-1] + p.bbox[:2],  # vertices
                np.full((contour.shape[0], 1), p.label)  # ID
            ])
            contours.append(contour)
    contours = np.concatenate(contours)
    
    return contours


def contours_to_polygons(
    x: ArrayLike,
    y: ArrayLike,
    ids: ArrayLike,
) -> gpd.GeoDataFrame:
    """
    Convert contour vertices into Shapely polygons.

    Parameters
    ----------
    x : ArrayLike of shape (N,)
        x-coordinates of contour vertices.
    y : ArrayLike of shape (N,)
        y-coordinates of contour vertices.
    ids : ArrayLike of shape (N,)
        Cell ID for each (x, y) vertex. Contiguous vertices share the same ID.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing Shapely polygons, indexed by unique cell ID.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ids = np.asarray(ids)

    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = np.arange(len(np.unique(ids)) + 1)
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        coords=np.stack([x, y]).T.copy(order='C'),
        offsets=(geometry_offset, part_offset),
    )

    indices = np.sort(np.unique(ids, return_index=True)[1])
    return gpd.GeoDataFrame(geometry=polygons, index=ids[indices])

def main(masks_file_path, results_dir: str):
    masks = tifffile.imread(masks_file_path, series=0, level=0, key=0)
    contours = masks_to_contours(masks)
    gdf = contours_to_polygons(contours[:, 0], contours[:, 1], contours[:, 2])
    output_file_path = Path(results_dir) / "segmentation_masks.parquet"
    gdf.to_parquet(
        output_file_path,
        write_covering_bbox=True,
        geometry_encoding="geoarrow"
    )

if __name__ == "__main__":
    masks_file_path = sys.argv[1]
    results_dir = Path(masks_file_path).parent

    main(masks_file_path=masks_file_path, results_dir=results_dir)

