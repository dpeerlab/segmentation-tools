# segmentation-tools

Segmentation Tools

Arguments:
	"-f", "--fixed_file", required=True, help="Directory path for Xenium data"
	"-m", "--moving_file", required=True, help="Input file path for IF image (e.g., .tiff)"
	"-o", "--output_dir", required=True, help="Output directory path"

	"-j", "--job_title", default="output", help="Job title for output directory"
	"-n", "--nuclei_channel_moving", type=int, help="DAPI channel index for moving image"
	"--series_fixed", type=int, default=0, help="Series index for fixed image (default: 0)"
	"--series_moving", type=int, default=0, help="Series index for moving image (default: 0)"

	"--high_res_level", type=int, default=0, help="Resolution level to use for high-res alignment (default: 0)"
	"--find_poorly_aligned_regions", type=bool, default=False, help="Whether to find poorly aligned regions (default: False)"
	"--apply_mirage_correction", type=bool, default=True, help="Whether to apply mirage correction to SIFT aligned images (default: False)"
	"--save_intermediate_outputs", type=bool, default=False, help="Whether to save intermediate outputs (default: False)"

Workflow:
Reads in moving file and converts to .tiff if needed using bioformats2raw and raw2ometiff
Determines the best low resolution levels for finding the SIFT alignment between the moving and fixed images (currently finds levels that have over 1500 pixels, currently returns the same level for both)
Load the downsampled moving and fixed images, normalize (CLAHE), and match histograms between the images to correct for any differences in contrast/intensity
Determine rotations and reflections to match the images (using SSIM) and then SIFT transformation to find a linear mapping between the images and apply them
Combine the downsampling/upsampling transformations with the SIFT transformation to allow for direct conversion between high res images
(Optional) Find poorly aligned regions in the image and draw boxes around them using an SSIM/brightness mask
Load the high resolution moving and fixed images, normalize (CLAHE), and match histograms between the images to correct for any differences in contrast/intensity
Warp the high resolution moving image to match the high resolution fixed image
(Optional) Run MIRAGE alignment on the DAPI channel (eventually want to get the transformation to apply to all channels
Downsample from high resolution to create OME pyramidal TIFF file
(Optional) Save poorly aligned regions overlays at high resolution
