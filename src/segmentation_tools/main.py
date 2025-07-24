from argparse import ArgumentParser
from segmentation_tools.pipelines import AlignmentPipeline, SegmentationPipeline
import sys


def main():
    parser = ArgumentParser(description="Segmentation CLI")
    subparsers = parser.add_subparsers(dest="command")  # no required=True

    # align
    p_align = subparsers.add_parser("align")
    p_align.add_argument("--job_title", default="output")
    p_align.add_argument("--if_file", required=True)
    p_align.add_argument("--xenium_dir", required=True)
    p_align.add_argument("--output_dir")
    p_align.add_argument("--nuclei_channel_if", type=int, default=1)
    p_align.add_argument("--membrane_channel_if", type=int, default=0)

    # segment
    # p_segment = subparsers.add_parser("segment")
    # p_segment.add_argument("--output_dir")

    # both
    # p_both = subparsers.add_parser("run")
    # p_both.add_argument("--if_file", required=True)
    # p_both.add_argument("--xenium_dir", required=True)
    # p_both.add_argument("--output_dir", default="output")
    # p_both.add_argument("--nuclei_channel_if", type=int, default=1)

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
