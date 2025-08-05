import sys
from argparse import ArgumentParser
from icecream import install

from segmentation_tools.pipelines import (AlignmentPipeline,
                                          SegmentationPipeline)


def main():
    # Install icecream for debugging
    install()
    # Set up argument parser
    parser = ArgumentParser(description="Segmentation CLI")
    subparsers = parser.add_subparsers(dest="command")

    # align
    p_align = subparsers.add_parser("align")
    p_align.add_argument("-f", "--fixed_file", required=True, help="Directory path for Xenium data")
    p_align.add_argument("-m", "--moving_file", required=True, help="Input file path for IF image (e.g., .tiff)")
    p_align.add_argument("-o", "--output_dir", required=True, help="Output directory path")

    p_align.add_argument("-j", "--job_title", default="output", help="Job title for output directory")
    p_align.add_argument("-n", "--nuclei_channel_moving", type=int, help="DAPI channel index for moving image")
    p_align.add_argument("--series_fixed", type=int, default=0, help="Series index for fixed image (default: 0)")
    p_align.add_argument("--series_moving", type=int, default=0, help="Series index for moving image (default: 0)")

    p_align.add_argument("--high_res_level", type=int, default=0, help="Resolution level to use for high-res alignment (default: 0)")
    p_align.add_argument("--find_poorly_aligned_regions", type=bool, default=False, help="Whether to find poorly aligned regions (default: False)")
    p_align.add_argument("--apply_mirage_correction", type=bool, default=True, help="Whether to apply mirage correction to SIFT aligned images (default: False)")

    p_align.add_argument("--save_intermediate_outputs", type=bool, default=False, help="Whether to save intermediate outputs (default: False)")

    if len(sys.argv) <= 1 or sys.argv[1] not in ("align", "segment", "run"):
        # inject "run" as default
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    if args.command == "align":
        AlignmentPipeline(**vars(args)).run()
    elif args.command == "segment":
        SegmentationPipeline(output_dir=args.output_dir).run()
    elif args.command == "run":
        align = AlignmentPipeline(**vars(args))
        align.run()
        segment = SegmentationPipeline(output_dir=args.output_dir)
        segment.run()


if __name__ == "__main__":
    main()
