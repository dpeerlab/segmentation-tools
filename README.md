# segmentation-tools

Example output:
/data1/peerd/moormana/data/segger/xenium_andrew_cornish/andrew_c_alignment

Inputs:
- Output root: this is where the intermediate and result files will go (andrew_c_alignment)
- Job Title - name of the subdir within the output root. (output-XETG00174__0064842__Region_1__20250514__154857)
- Moving file - this is file containing the image the will be aligned (can be .tiff, .nd2, .mrxs...) -> bioformats2raw library should take in most microscopy formats and raw2ometiff will convert the output from the first package to an OME TIFF file
- Fixed file - this if the file containg the image that the moving file will be aligned too (for now we've only done xenium so it should be morpholoogy_focus subdir)
- HIGH_RES_LEVEL - highest resolution image to work with (can usually be 0)
- FIXED_DAPI_CHANNEL - 0
- MOVING_DAPI_CHANNEL - channel that contains the DAPI image

1. Step 0 - set up directories:
	Args: --output-root, --job-title
	- Create Directory at OUTPUT_ROOT
	- Create a subidirectory at OUTPUT_ROOT/JOB_TITLE
- At the end should have a dir and a subdir
  
1. Step 1 - convert moving and fixed files to an OME TIFF for fixed and moving separately:
	Args: --input-path, --output-root (change name to avoid confusion) OUTPUT_ROOT/JOB_TITLE, --prefix (prepended to the output path of the converted file e.g. fixed, right now is just the name of the resulting file)
	- Calls the bioformats2raw library which takes in different microscopy formats and converts a zarr file, raw2ometiff takes that zarr file and converts to a tiff file
- At the ened should have a moving.tiff and a fixed.tiff in the OUPTUT_ROOT/JOB_TITLE
  
1. Step 2 - Find optimal levels for SIFT
	- Args: --moving-file (should be under ${OUTPUT_ROOT}/${JOB_TITLE}/.checkpoints/moving.tiff), --fixed-file, --k-min (minimum keypoints), --k-max (maximum keypoints)
	Checks the number of keypoints that are found by SIFT and takes in a min and a mask. Loops through all the levels and finds the coarsest one that is between the bounds
At the end should have a new file called `optimal_sift_levels.txt` in the .checkpoints folder

1. Step 3 - Preprocess images
	Separate for fixed and moving
   - Args: --input-file-path (path to either fixed of moving file - should be in .checkpoints folder), --dapi-channel-moving (should be specified at the beginning of the pipeline), --level (read in from the `optimal_sift_levels.txt` file), --output-file-path (path to the checkpoints dir, right now si set to SUB_DIR/ds_fixed_dapi_filtered_level_{SIFT_LEVEL}.npy
	Performs normalization

2. Step 4 - Find SIFT homography matrix
	Inputs: --moving-file-path (path to the downsampled moving file (should be specified in the call to the process above)), --fixed-file-path (same as moving), --high-res-level (the high res level specified at the beginning of the pipeline), --original-moving-file-path (path to the tiff file we got from step 1), --original-fixed-file-path (path to the tiff file we got from step 1)
	Saves the homography matrix to .checkpoints/linear_transform.npy

3. Step 4.5 Preprocess the high res images
	Same as step 3, but we pass in an extra argument (--filter), this also runs the adaptive Otsu threshold filtering

4. Step 5: Warp the high res image
   Inputs: --moving-file-path: Path to the high res movig filtered file, --transform-file-path: reads in the transfrom file path that was outputted by step 4
   - Warps the image and outputs to a file "moving_dapi_linear_warped.npy"
   - Could possibly run using the cucim library (GPU), would have to check speed improvement and memory usage to determine if it's worth

5. Step 6: MIRAGE
	Inputs: --warped-file-path: path to the warped file from the previous process, --fixed-file-path: path to the high res fixed filtered file from step 4.5, --batch-size (use the recommended 1024), --learning-rate (use the recommended 0.012575)
	Runs mirage and outputs the transform to checkpoints/mirage_transform.npy

6. Step 7: Apply warp to channels
	Inputs: --moving-file-path: file path for the moving.tiff file from step 1, --high-res-level (get this from start of the pipeline), if it's lower for any reason it will only warp channels at that level and below
	Normalize each of the channels and applies the combined mirage, linear sift transform. Downsamples from the top of the pyramid so that we have all resolution levels
	Outputs to results folder: results/moving_complete_transform.ome.tiff

Optional step:
	Calculate SSIM before and after MIRAGE and output SSIM plots
	-- outputs to checkpoints dir


8. Step 8: Segment with CellPose
   - Inputs: --warped-moving-file-path (this is the tiff file outputted from the last step), --dapi-channel (should be the same as the one from the beginning), --membrane-channel (should actually also be specified by the user, right now it's hardcoded and it shouldn't be)
   - Runs CellPose with (membrane + DAPI), DAPI alone, and membrane alone (eventually will combine the three in a smart way to ge the best segmentation)
   - outputs a masks npy image, and the cell probabilities

9. Step 9: Convert masks to GeoDataFrame
    - Does what it says figure it out Matt I'm tired
    - outputs to a parquet file (should probably indicate that it's a geo parquet)


