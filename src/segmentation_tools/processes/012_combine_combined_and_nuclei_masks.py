import geopandas as gpd
import pandas as pd
import argparse
from loguru import logger
from pathlib import Path
from segmentation_tools.utils.profiling import profile_step, profile_block


def join_masks(combined_masks: gpd.GeoDataFrame, nuclei_masks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Joins combined and nuclei masks into a single GeoDataFrame."""
    logger.info("Starting spatial join of combined and nuclei masks.")
    intersecting_join = gpd.sjoin(
        nuclei_masks.reset_index(names=['original_nuclei_index']), # Create a temporary index column
        combined_masks,
        how='left',
        predicate='intersects'
    )
    intersecting_nuclei_indices = intersecting_join[intersecting_join['index_right'].notna()]['original_nuclei_index'].unique()
    non_overlapping_nuclei_gdf = nuclei_masks[~nuclei_masks.index.isin(intersecting_nuclei_indices)]
    final_gdf = pd.concat([combined_masks, non_overlapping_nuclei_gdf], ignore_index=True)
    logger.info("Completed spatial join of combined and nuclei masks.")
    return final_gdf


def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Combined combined and nuclei masks into a single GeoDataFrame."
    )

    parser.add_argument(
        "--combined-masks",
        required=True,
        type=str,
        help="Path to the combined masks file.",
    )

    parser.add_argument(
        "--nuclei-masks",
        required=True,
        type=str,
        help="Path to the nuclei masks file.",
    )

    return parser.parse_args()

@profile_step("012 Combine Combined and Nuclei Masks")
def main():
    args = parse_arguments()

    combined_masks_path = args.combined_masks
    nuclei_masks_path = args.nuclei_masks

    final_gdf_path = Path(combined_masks_path).parent / "final_combined_nuclei_masks.parquet"
    if final_gdf_path.exists():
        logger.info(f"Final combined nuclei masks already exist at {final_gdf_path}. Skipping.")
        return 0

    with profile_block("Load combined masks"):
        combined_gdf = gpd.read_parquet(combined_masks_path)
    logger.info(f"Combined masks: {len(combined_gdf)} polygons from {combined_masks_path}")

    with profile_block("Load nuclei masks"):
        nuclei_gdf = gpd.read_parquet(nuclei_masks_path)
    logger.info(f"Nuclei masks: {len(nuclei_gdf)} polygons from {nuclei_masks_path}")

    with profile_block("Spatial join"):
        final_gdf = join_masks(combined_gdf, nuclei_gdf)
    logger.info(f"Final merged mask: {len(final_gdf)} polygons")

    final_gdf.to_parquet(final_gdf_path, index=False)
    logger.info(f"Saved to {final_gdf_path}")

if __name__ == "__main__":
    main()