from pathlib import Path
from pydantic import BaseModel, Field, validator, PrivateAttr
import tifffile
from segmentation_tools.logger import logger
from segmentation_tools.utils.convert_image_utils import convert_to_tiff_if_needed
import segmentation_tools.utils.image_utils as image_utils
import segmentation_tools.utils.sift_alignment_utils as sift_utils
import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import AffineTransform
from skimage.metrics import structural_similarity as ssim


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

    _intermediates_dir: Path = PrivateAttr()

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
        self._intermediates_dir = self.output_dir / "intermediates"
        self._intermediates_dir.mkdir(parents=True, exist_ok=True)

    ### ALIGNMENT HELPER FUNCTIONS ###

    def _determine_alignment_levels(self):
        return sift_utils.get_best_common_level(
            self.if_file,
            self.xenium_file,
            min_size=1500,
        )

    def _load_and_normalize_images(self, level_moving, level_fixed):
        # Read and normalize IF images at the specified levels
        dapi_img_if_ds = tifffile.imread(
            self.if_file,
            series=self.series_if,
            key=self.nuclei_channel_if,
            level=level_moving,
        )
        dapi_img_if_high_res = tifffile.imread(
            self.if_file,
            series=self.series_if,
            key=self.nuclei_channel_if,
            level=self.high_res_level,
        )
        membrane_img_if_high_res = tifffile.imread(
            self.if_file,
            series=self.series_if,
            key=self.membrane_channel_if,
            level=self.high_res_level,
        )

        # Read and normalize Xenium image at the specified levels
        dapi_img_xenium_ds = tifffile.imread(
            self.xenium_file, series=self.series_xenium, level=level_fixed
        )

        # Normalize images
        normalized_if_dapi_ds = image_utils.normalize(
            dapi_img_if_ds, return_float=False
        )
        normalized_if_dapi_high_res = image_utils.normalize(
            dapi_img_if_high_res, return_float=False
        )
        normalized_if_membrane_high_res = image_utils.normalize(
            membrane_img_if_high_res, return_float=False
        )
        normalized_xenium_dapi_ds = image_utils.normalize(
            dapi_img_xenium_ds, return_float=False
        )

        return (
            normalized_if_dapi_ds,
            normalized_if_membrane_high_res,
            normalized_if_dapi_high_res,
            normalized_xenium_dapi_ds,
        )

    ### RUN FUNCTIONS ###
    def run(self):
        logger.info(
            f"Starting alignment pipeline at high resolution level {self.high_res_level}"
        )

        # Convert IF file to TIFF if needed
        self.if_file = convert_to_tiff_if_needed(
            input_file_path=self.if_file,
            intermediates_dir=self._intermediates_dir,
        )

        # Get downsampled levels for alignment
        level_fixed, level_moving = self._determine_alignment_levels()
        logger.info(
            f"Using levels for downsampled alignment - Fixed: {level_fixed}, Moving: {level_moving}"
        )

        # Load and normalize images at downsampled levels
        (
            normalized_if_dapi_ds,
            normalized_if_membrane_high_res,
            normalized_if_dapi_high_res,
            normalized_xenium_dapi_ds,
        ) = self._load_and_normalize_images(level_moving, level_fixed)

        # Get transforms for downsampled images
        tm_moving_ds = image_utils.get_level_transform(
            self.if_file, level_to=level_moving, level_from=self.high_res_level
        )
        tm_fixed_ds = image_utils.get_level_transform(
            self.xenium_file, level_to=level_fixed, level_from=self.high_res_level
        )

        # Find SIFT transform between downsampled images
        tm_sift = sift_utils.find_best_sift(
            mvg_img=normalized_if_dapi_ds, fxd_img=normalized_xenium_dapi_ds
        )

        # Warp downsampled
        logger.info("Warping IF DAPI at downsampled resolution")
        warped_if = skimage.transform.warp(
            normalized_if_dapi_ds,
            tm_sift.inverse,
            output_shape=normalized_xenium_dapi_ds.shape,
            preserve_range=True,
        )

        image_utils.save_visualization_overlay(
            image_fixed=normalized_xenium_dapi_ds,
            image_moving=warped_if,
            output_file_path=self._intermediates_dir / "dapi_downsampled_overlay.png",
        )

        # Get transform for high resolution
        tm_combined = tm_moving_ds + tm_sift.inverse + tm_fixed_ds.inverse

        # Warp DAPI image at high resolution
        dapi_if_warped_high_res = skimage.transform.warp(
            normalized_if_dapi_high_res,
            tm_combined,
            output_shape=image_utils.get_shape_at_level(
                self.xenium_file, level=self.high_res_level
            ),
            preserve_range=True,
        )

        # Save warped warped stacked image to tiff
        image_utils.save_image(
            image=dapi_if_warped_high_res,
            output_file_path=self._intermediates_dir / "dapi_if_warped_high_res.tiff",
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
            [dapi_if_warped_high_res, membrane_if_warped_high_res], axis=-1
        )

        # Save warped warped stacked image
        image_utils.save_image(
            image=if_warped_stacked_high_res,
            output_file_path=self._intermediates_dir / "stacked_dapi_membrane.tiff",
            description="High-res warped IF stacked",
        )

        dapi_img_xenium_high_res = tifffile.imread(
            self.xenium_file,
            series=self.series_xenium,
            level=self.high_res_level,
        )

        normalized_dapi_xenium_high_res = image_utils.normalize(
            dapi_img_xenium_high_res, return_float=False
        )

        # Save warped warped stacked image
        image_utils.save_image(
            image=normalized_dapi_xenium_high_res,
            output_file_path=self._intermediates_dir / "normalized_dapi_xenium_high_res.tiff",
            description="High-res noramlized DAPI Xenium",
        )


        return
