import argparse
import time
import logging

from nuclei_segmentation.pipeline import segment_and_classify, segment_and_classify_parallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Segment and classify nuclei in microscopy images."
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save output image")
    parser.add_argument(
        "--method",
        choices=["watershed", "deep_learning"],
        default="watershed",
        help="Segmentation method (default: watershed)",
    )
    # Possible other options for tile size and overlap
    # parser.add_argument(
    #     "--tile_size",
    #     type=int,
    #     default=2000,
    #     help="Tile size for large image segmentation",
    # )
    # parser.add_argument(
    #     "--overlap",
    #     type=int,
    #     default=200,
    #     help="Tile overlap for parallel processing",
    # )
    args = parser.parse_args()

    if args.method == "watershed":
        segment_and_classify_parallel(args.input, args.output)
    else:  # args.method == "deep_learning"
        segment_and_classify(args.input, args.output, method='deep_learning')
    
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Processing time: {elapsed:.2f} seconds")