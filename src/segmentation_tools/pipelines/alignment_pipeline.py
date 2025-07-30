import gc
import os
import shutil
from pathlib import Path

import numpy as np
import skimage
import tifffile
from pydantic import BaseModel, Field, PrivateAttr, validator

import segmentation_tools.utils.alignment_utils as alignment_utils
import segmentation_tools.utils.image_utils as image_utils
from segmentation_tools.logger import logger
from segmentation_tools.utils.convert_image_utils import \
    convert_to_tiff_if_needed


class AlignmentPipeline(BaseModel):
    if_file: Path = Field(..., description="Path to the input IF image")
    xenium_dir: Path = Field(..., description="Path to the Xenium output directory")
    xenium_file: Path = Field(default=None, description="Path to the Xenium file")
    output_dir: Path = Field(..., description="Path to store outputs")
    job_title: str = Field(
        default="output", description="Title for the job, used in output naming"
    )

    nuclei_channel_if: int = Field(
        1, description="DAPI channel index for IF (default: 1)"
    )
    membrane_channel_if: int = Field(
        0, description="Membrane channel index for IF (default: 0)"
    )

    series_if: int = Field(0, description="Series index for IF (default: 0)")
    series_xenium: int = Field(0, description="Series index for Xenium (default: 0)")

    high_res_level: int = Field(
        0, description="High resolution level for alignment (default: 0)"
    )

    _processed_tiff_dir: Path = PrivateAttr()
    _poor_regions_dir: Path = PrivateAttr()
    _rotations_dir: Path = PrivateAttr()
    _overlay_dir: Path = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_dirs()
        (self.output_dir / ".alignment_done").unlink(missing_ok=True)

        if self.xenium_file is None:
            self.xenium_file = (
                self.xenium_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
            )

    @validator("if_file")
    def check_if_file_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Input file does not exist: {v}")
        return v

    @validator("xenium_dir")
    def check_xenium_dir_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Xenium output directory does not exist: {v}")
        return v

    def _setup_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.output_dir / self.job_title

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._processed_tiff_dir = self.output_dir / "processed_tiffs"
        self._processed_tiff_dir.mkdir(parents=True, exist_ok=True)

        self._poor_regions_dir = self.output_dir / "poorly_aligned_regions"
        self._poor_regions_dir.mkdir(parents=True, exist_ok=True)

        self._rotations_dir = self.output_dir / "rotations"
        self._rotations_dir.mkdir(parents=True, exist_ok=True)

        self._overlay_dir = self.output_dir / "overlays"
        self._overlay_dir.mkdir(parents=True, exist_ok=True)
        return

    ### ALIGNMENT HELPER FUNCTIONS ###

    def _determine_alignment_levels(self):
        return alignment_utils.get_best_common_level(
            self.if_file,
            self.xenium_file,
            min_size=2000,
        )

    def _garbage_collect_objects(self, objects):
        """
        Helper function to delete one or more objects and force garbage collection.

        Args:
            objects: A single object or an iterable of objects to delete.
        """
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        for obj in objects:
            del obj
        gc.collect()

    ### RUN FUNCTIONS ###
    def run(self):
        logger.info(
            f"Starting alignment pipeline at high resolution level {self.high_res_level}"
        )

        logger.info(f"Input IF file: {self.if_file}\nXenium file: {self.xenium_file}")

        # Convert IF file to TIFF if needed
        self.if_file = convert_to_tiff_if_needed(
            input_file_path=self.if_file,
            intermediates_dir=self._processed_tiff_dir,
        )

        # Get downsampled levels for alignment
        level_fixed, level_moving = self._determine_alignment_levels()
        logger.info(
            f"Using levels for downsampled alignment - Fixed: {level_fixed}, Moving: {level_moving}"
        )

        # Load and normalize images at downsampled levels
        dapi_img_if_ds = tifffile.imread(
            self.if_file,
            series=self.series_if,
            key=self.nuclei_channel_if,
            level=level_moving,
        )
        # Read and normalize Xenium image at the specified levels
        dapi_img_xenium_ds = tifffile.imread(
            self.xenium_file, series=self.series_xenium, level=level_fixed
        )

        # Normalize images
        normalized_if_dapi_ds = image_utils.normalize(
            dapi_img_if_ds, return_float=False
        )
        normalized_xenium_dapi_ds = image_utils.normalize(
            dapi_img_xenium_ds, return_float=False
        )

        

        # TODO: DO MATCHING BASED ON LOCAL TILES
        normalized_if_dapi_ds, normalized_xenium_dapi_ds = (
            alignment_utils.match_image_histograms(
                normalized_if_dapi_ds,
                normalized_xenium_dapi_ds,
            )
        )

        exit()

        # Save normalized images to tiff
        image_utils.save_image(
            image=normalized_if_dapi_ds,
            output_file_path=self._processed_tiff_dir / "normalized_dapi_if_ds.tiff",
            description="Downsampled normalized DAPI IF",
        )

        # Find SIFT transform between downsampled images
        tm_sift = alignment_utils.find_best_sift(
            mvg_img=normalized_if_dapi_ds,
            fxd_img=normalized_xenium_dapi_ds,
            save_img_dir=self._rotations_dir,
            draw_matches=False,
        )

        # Warp downsampled
        logger.info("Warping IF DAPI at downsampled resolution")
        warped_if = skimage.transform.warp(
            normalized_if_dapi_ds,
            tm_sift,
            output_shape=normalized_xenium_dapi_ds.shape,
            preserve_range=True,
        )

        # Save warped warped stacked image to tiff
        image_utils.save_image(
            image=warped_if,
            output_file_path=self._processed_tiff_dir / "dapi_if_warped_ds.tiff",
            description="Downsampled warped DAPI",
        )

        # Save warped warped stacked image to tiff
        image_utils.save_image(
            image=normalized_xenium_dapi_ds,
            output_file_path=self._processed_tiff_dir / "dapi_xenium_ds.tiff",
            description="Downsampled Xenium DAPI",
        )

        poorly_aligned_regions = alignment_utils.find_poorly_aligned_regions(
            img1=warped_if,
            img2=normalized_xenium_dapi_ds,
            win_size=11,
            min_brightness_factor=0.15,
            min_area_factor=5e-5,
            ssim_bounds=(0.0, 0.6),
            output_file_path=self._poor_regions_dir / "poorly_aligned_regions",
        )

        image_utils.save_full_overlay(
            image_fixed=normalized_xenium_dapi_ds,
            image_moving=warped_if,
            boxes=poorly_aligned_regions,
            output_file_path=self._poor_regions_dir / "dapi_downsampled_overlay",
        )

        # Free memory
        self._garbage_collect_objects(
            [
                normalized_if_dapi_ds,
                normalized_xenium_dapi_ds,
                warped_if,
            ]
        )

        # Get transform for high resolution
        tm_combined = (
            image_utils.get_level_transform(
                self.xenium_file, level_to=level_moving, level_from=self.high_res_level
            ).params
            @ tm_sift.params
            @ image_utils.get_level_transform(
                self.xenium_file, level_to=self.high_res_level, level_from=level_fixed
            ).params
        )

        # Warp DAPI image at high resolution
        normalized_dapi_if_warped_high_res = skimage.transform.warp(
            normalized_if_dapi_high_res,
            tm_combined,
            output_shape=image_utils.get_shape_at_level(
                self.xenium_file, level=self.high_res_level
            ),
            preserve_range=True,
        )

        dapi_img_xenium_high_res = tifffile.imread(
            self.xenium_file,
            series=self.series_xenium,
            level=self.high_res_level,
        )

        normalized_dapi_xenium_high_res = image_utils.normalize(
            dapi_img_xenium_high_res, return_float=False
        )

        # Save warped  stacked image
        image_utils.save_image(
            image=normalized_dapi_xenium_high_res,
            output_file_path=self._processed_tiff_dir
            / "normalized_dapi_xenium_high_res.tiff",
            description="High-res normalized DAPI Xenium",
        )

        transformed_poorly_aligned_regions = image_utils.transform_polygons_to_high_res(
            polygons=poorly_aligned_regions,
            transform=image_utils.get_level_transform(
                self.xenium_file, level_to=level_fixed, level_from=self.high_res_level
            ),
        )


        normalized_dapi_xenium_high_res, normalized_dapi_if_warped_high_res = (
            alignment_utils.match_image_histograms(
                normalized_dapi_xenium_high_res,
                normalized_dapi_if_warped_high_res,
            )
        )

        # Moving on top of fixed (default behavior)
        image_utils.save_full_overlay(
            image_fixed=normalized_dapi_xenium_high_res,
            image_moving=normalized_dapi_if_warped_high_res,
            boxes=transformed_poorly_aligned_regions,
            output_file_path=self._poor_regions_dir / "dapi_high_res_overlay_moving_on_top_overlay.png",
            fixed_on_top=False,
        )

        # Fixed on top of moving
        image_utils.save_full_overlay(
            image_fixed=normalized_dapi_xenium_high_res,
            image_moving=normalized_dapi_if_warped_high_res,
            boxes=transformed_poorly_aligned_regions,
            output_file_path=self._poor_regions_dir / "dapi_high_res_overlay_fixed_on_top_overlay.png",
            fixed_on_top=True,
        )

        # Save cropped fixed_on_top image
        image_utils.save_cropped_overlays(
            image_fixed=normalized_dapi_xenium_high_res,
            image_moving=normalized_dapi_if_warped_high_res,
            boxes=transformed_poorly_aligned_regions,
            output_file_path=self._poor_regions_dir / "dapi_high_res_cropped_overlay_fixed_on_top_overlay",
            fixed_on_top=True,
        )

        # Save cropped fixed_on_top image
        image_utils.save_cropped_overlays(
            image_fixed=normalized_dapi_xenium_high_res,
            image_moving=normalized_dapi_if_warped_high_res,
            boxes=transformed_poorly_aligned_regions,
            output_file_path=self._poor_regions_dir / "dapi_high_res_cropped_overlay_fixed_on_top_overlay",
            fixed_on_top=False,
        )

        # Free memory
        self._garbage_collect_objects(
            [
                dapi_img_xenium_high_res,
                normalized_dapi_xenium_high_res,
            ]
        )

        # Save warped stacked image to tiff
        image_utils.save_image(
            image=normalized_dapi_if_warped_high_res,
            output_file_path=self._processed_tiff_dir / "dapi_if_warped_high_res.tiff",
            description="High-res warped DAPI",
        )

        # Warp membrane image at high resolution
        membrane_if_warped_high_res = skimage.transform.warp(
            normalized_if_membrane_high_res,
            tm_combined,
            output_shape=image_utils.get_shape_at_level(
                self.xenium_file, level=self.high_res_level
            ),
            preserve_range=True,
        )

        # Stack warped DAPI and membrane images to tiff
        if_warped_stacked_high_res = np.stack(
            [normalized_dapi_if_warped_high_res, membrane_if_warped_high_res], axis=-1
        )

        # Save warped warped stacked image
        image_utils.save_image(
            image=if_warped_stacked_high_res,
            output_file_path=self._processed_tiff_dir / "stacked_dapi_membrane.tiff",
            description="High-res warped IF stacked",
        )

        # Free memory
        self._garbage_collect_objects(
            [
                normalized_dapi_if_warped_high_res,
                membrane_if_warped_high_res,
                if_warped_stacked_high_res,
            ]
        )
