from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr
import tifffile
import numpy as np
import imageio.v3 as iio
import cellpose
from segmentation_tools.logger import logger
import segmentation_tools.utils as utils
from skimage.exposure import rescale_intensity
from cellpose import models, core, io, plot
import torch

class SegmentationPipeline(BaseModel):
    output_dir: Path = Field(..., description="Path to store outputs")

    _intermediates_dir: Path = PrivateAttr()
    _if_dapi_img_aligned: np.ndarray = PrivateAttr()

    def model_post_init(self, __context) -> None:
        (self.output_dir / ".segmentation_done").unlink(missing_ok=True)
        self._intermediates_dir = self.output_dir / "intermediates"
        self._if_dapi_img_aligned = self._intermediates_dir / "if_dapi_aligned.npy"
        return

    def _is_gpu_availble(self):
        return torch.cuda.is_available()
            

    def _run_cellpose(self):
        """
        Run Cellpose segmentation on the aligned IF DAPI image.
        """
        logger.info("Running Cellpose segmentation")

        if not self._is_gpu_availble():
            raise RuntimeError("No GPU detected. Cellpose will run on CPU, which may be slow.")

        logger.info(f"PyTorch detected GPU: {torch.cuda.get_device_name(0)}")

        model = models.CellposeModel(gpu=True)
        masks, _, _ = model.eval(
            self._if_dapi_img_aligned,
            diameter=None,
            channels=[0, 0],
            normalize=True,
            resample=True,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
        output_path = self._intermediates_dir / "cellpose_masks.tiff"
        tifffile.imwrite(output_path, masks.astype(np.uint16))
        logger.info(f"Cellpose segmentation masks saved to {output_path}")

        mask_png = rescale_intensity(masks, out_range=(0, 255)).astype(np.uint8)
        mask_png_path = self._intermediates_dir / "cellpose_masks.png"
        iio.imwrite(mask_png_path, mask_png)
        logger.info(f"Cellpose segmentation masks saved as PNG to {mask_png_path}")
        return

    def run(self):
        logger.info("Starting segmentation pipeline")
        if not (self.output_dir / ".alignment_done").exists():
            raise RuntimeError("Alignment must be completed before segmentation.")
        
        if_dapi_aligned_path = self._intermediates_dir / "if_dapi_aligned.npy"
        if not if_dapi_aligned_path.exists():
            raise FileNotFoundError(f"Aligned IF DAPI image not found: {if_dapi_aligned_path}")
        else:
            self._if_dapi_img_aligned = np.load(self._intermediates_dir / "if_dapi_aligned.npy")
            logger.info(f"Loaded aligned IF DAPI image from {if_dapi_aligned_path}")
        
        self._run_cellpose()
        
        

