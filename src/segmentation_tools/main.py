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
    p_align.add_argument("--job_title", default="output")
    p_align.add_argument("--if_file", required=True)
    p_align.add_argument("--xenium_dir", required=True)
    p_align.add_argument("--output_dir")
    p_align.add_argument("--nuclei_channel_if", type=int, default=1)
    p_align.add_argument("--membrane_channel_if", type=int, default=0) 
    p_align.add_argument(
        "--high_res_level",
        type=int,
        default=0,
        help="Resolution level to use for high-res alignment (default: 0)",
    )

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
