from skimage.transform import AffineTransform, ProjectiveTransform
from typing import List
import geopandas as gpd
import pandas as pd
import scanpy as sc
import scipy as sp
import numpy as np
import ome_types
import tifffile
import shapely
import skimage
import cv2
import os


def get_image_transform(
    filepath: os.PathLike,
    level: int = 0,
):
    # Read twice for multi-file pyramid case
    tf = tifffile.TiffFile(filepath)
    md = ome_types.from_xml(tf.ome_metadata).images[0].pixels
    tf = tifffile.TiffFile(filepath, is_ome=False)
    if tf.series:
        lvl = tf.series[0].levels[level]
    else:
        lvl = tf.levels[level]
    dims = dict(zip(lvl.axes, lvl.shape))
    xres = (md.size_x / dims["X"]) * md.physical_size_x
    yres = (md.size_y / dims["Y"]) * md.physical_size_y
    return AffineTransform(scale=(xres, yres))


def get_crop_transform(
    xmin: float,
    ymin: float,
):
    return AffineTransform(translation=(-xmin, -ymin))


def get_level_transform(
    filepath: os.PathLike,
    level_to: int,
    level_from: int = 0,
):
    tf = tifffile.TiffFile(filepath)
    lvl = tf.series[0].levels[level_from]
    from_dims = dict(zip(lvl.axes, lvl.shape))
    lvl = tf.series[0].levels[level_to]
    to_dims = dict(zip(lvl.axes, lvl.shape))
    scale = to_dims["X"] / from_dims["X"], to_dims["Y"] / from_dims["Y"]
    return AffineTransform(scale=scale)


def get_SIFT_homography(
    img_fxd,  # Reference image to align to
    img_mvg,  # Image to transform
    min_match_count: int = 10,
):
    FLANN_INDEX_KDTREE = 1

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # Find the keypoints and descriptors with SIFT
    kp_fxd, dsc_fxd = sift.detectAndCompute(img_fxd, None)
    kp_mvg, dsc_mvg = sift.detectAndCompute(img_mvg, None)
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsc_fxd, dsc_mvg, k=2)
    # Store all the good matches as per Lowe's ratio test.
    good_matches = list(filter(lambda m: m[0].distance < 0.7 * m[1].distance, matches))
    # Compute 3x3 homography matrix
    if len(good_matches) >= min_match_count:
        fxd_pts = np.array([kp_fxd[m.queryIdx].pt for m, _ in good_matches])
        mvg_pts = np.array([kp_mvg[m.trainIdx].pt for m, _ in good_matches])
        M, mask = cv2.findHomography(mvg_pts, fxd_pts, cv2.RANSAC, 5.0)
        return ProjectiveTransform(M)
    else:
        msg = (
            f"Insufficient no. matches detected to perform image "
            f"registration: {len(good_matches)} matches"
        )
        raise AssertionError(msg)


def map_image_bounds(
    img: np.ndarray,
    transform: ProjectiveTransform,
):
    # Numpy is y, x and transforms are x, y ordered
    bounds = shapely.box(0, 0, *img.shape[::-1])
    xmin, ymin, xmax, ymax = shapely.transform(bounds, transform).bounds

    shape = np.ceil(ymax - ymin), np.ceil(xmax - xmin)
    translation = AffineTransform(translation=(-xmin, -ymin))

    return translation, shape


def get_polygons_from_xy(
    boundaries: pd.DataFrame,
    x: str,
    y: str,
    label: str,
) -> gpd.GeoDataFrame:
    """
    Convert boundary coordinates from a cuDF DataFrame to a GeoDataFrame of
    polygons.

    Parameters
    ----------
    boundaries : pd.DataFrame
        A DataFrame containing the boundary data with x and y coordinates
        and identifiers.
    x : str
        The name of the column representing the x-coordinate.
    y : str
        The name of the column representing the y-coordinate.
    label : str
        The name of the column representing the cell or nucleus label.


    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the polygons created from the boundary
        coordinates.
    """
    # Polygon offsets in coords
    ids = boundaries[label].values
    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = np.arange(len(np.unique(ids)) + 1)

    # Convert to GeoSeries of polygons
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        coords=boundaries[[x, y]].values.copy(order="C"),
        offsets=(geometry_offset, part_offset),
    )
    gs = gpd.GeoDataFrame(geometry=polygons, index=np.unique(ids))

    return gs


def masks_to_contours(masks: np.ndarray) -> np.ndarray:
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
    for i, p in enumerate(props):
        # Get largest contour with label
        lbl_contours = cv2.findContours(
            np.pad(p.image, 1).astype("uint8"),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        contour = sorted(lbl_contours, key=lambda c: c.shape[0])[-1]
        if contour.shape[0] > 2:
            contour = np.hstack(
                [
                    np.squeeze(contour)[:, ::-1] + p.bbox[:2],  # vertices
                    np.full((contour.shape[0], 1), i),  # ID
                ]
            )
            contours.append(contour)
        else:
            print("error")
    contours = np.concatenate(contours)

    return contours


def contours_to_polygons(
    contours: np.ndarray,
    transform: skimage.transform.ProjectiveTransform = None,
) -> gpd.GeoSeries:
    """
    Convert contour vertices into Shapely polygons.

    Parameters
    ----------
    contours : np.ndarray
        Array of shape (3, N) where rows are x, y, and cell ID.
    transform : ProjectiveTransform, optional
        Transformation to apply to polygon coordinates.

    Returns
    -------
    gpd.GeoSeries
        A GeoSeries of polygons indexed by cell ID.
    """
    # Check shape of contours
    contours = np.array(contours)
    if contours.shape[0] != 3 and contours.shape[1] == 3:
        contours = contours.T
    # Convert to GeoSeries of Shapely polygons
    ids = contours[2]
    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = np.arange(len(np.unique(ids)) + 1)
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        coords=contours[:2].T.copy(order="C"),
        offsets=(geometry_offset, part_offset),
    )
    if transform:
        polygons = shapely.transform(polygons, transform)
    return gpd.GeoSeries(polygons, index=np.unique(ids))


def anndata_from_transcripts(
    transcripts: pd.DataFrame,
    cell_label: str,
    gene_label: str,
    coordinate_labels: List[str] = None,
):
    """
    Create an AnnData object from a transcript-level DataFrame.

    Parameters
    ----------
    transcripts : pd.DataFrame
        DataFrame containing transcript-level information with at least cell and
        gene labels.
    cell_label : str
        Column name in `transcripts` specifying cell identifiers.
    gene_label : str
        Column name in `transcripts` specifying gene identifiers.
    coordinate_labels : list of str, optional
        List of column names specifying spatial coordinates (e.g., ['x', 'y']).
        If provided, spatial centroids are computed and stored in `obsm['X_spatial']`.

    Returns
    -------
    adata : sc.AnnData
        AnnData object with cells as observations and genes as variables. Spatial
        coordinates are stored in `obsm['X_spatial']` if `coordinate_labels` are
        provided.
    """
    # Feature names to indices
    ids_cell, labels_cell = pd.factorize(transcripts[cell_label])
    ids_gene, labels_gene = pd.factorize(transcripts[gene_label])

    # Remove NaN values
    mask = ids_cell >= 0
    ids_cell = ids_cell[mask]
    ids_gene = ids_gene[mask]

    # Sort row index
    order = np.argsort(ids_cell)
    ids_cell = ids_cell[order]
    ids_gene = ids_gene[order]

    # Build sparse matrix
    X = sp.sparse.coo_matrix(
        (
            np.ones_like(ids_cell),
            np.stack([ids_cell, ids_gene]),
        ),
        shape=(len(labels_cell), len(labels_gene)),
    ).tocsr()

    # To AnnData
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=labels_cell.astype(str)),
        var=pd.DataFrame(index=labels_gene),
    )
    adata.raw = adata.copy()

    # Add spatial coords
    if coordinate_labels is not None:
        coords = transcripts[coordinate_labels]
        centroids = coords.groupby(transcripts[cell_label]).mean()
        idx = adata.obs.index.astype(transcripts[cell_label].dtype)
        adata.obsm["X_spatial"] = centroids.loc[idx].values

    return adata
