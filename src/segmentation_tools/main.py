from argparse import ArgumentParser
from segmentation_tools.pipeline import SegmentationPipeline


def main():
    parser = ArgumentParser(description="Run the segmentation pipeline")
    parser.add_argument("--if_file", required=True, help="Path to IF image file")
    parser.add_argument(
        "--xenium_dir", required=True, help="Path to Xenium output directory"
    )
    parser.add_argument(
        "--output_dir", default="output", help="Directory to store outputs"
    )
    parser.add_argument(
        "--dapi_channel", type=int, default=0, help="DAPI channel index (default: 0)"
    )

    args = parser.parse_args()

    pipeline = SegmentationPipeline(
        if_file=args.if_file,
        xenium_dir=args.xenium_dir,
        output_dir=args.output_dir,
        dapi_channel=args.dapi_channel,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
