#!/usr/bin/env python
"""
run_inference.py

CLI to run bomb detection on a batch of WAV files, writing into a pre-created --output-dir.
"""
# TODO: Get it to print current batch num/total batch num at the start of the tqdm file progress bar

import argparse
import logging
from pathlib import Path

from tqdm.auto import tqdm
import pandas as pd

from config import MODEL_DIR, INPUT_DIR
from inference.bomb_detector import BombDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bomb detection on a batch of WAV files."
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help="Path to saved TensorFlow model."
    )
    parser.add_argument(
        "--input-dir",
        default=str(INPUT_DIR),
        help="Directory containing raw WAV files."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pre-created output directory (batch_runner takes care of this)."
    )
    parser.add_argument(
        "--batch-file",
        required=True,
        help="Text file listing WAV filenames (one per line)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    # assume batch_runner already created/validated this

    batch_list = Path(args.batch_file).read_text().splitlines()
    if not batch_list:
        logging.warning("Batch file %s is empty.", args.batch_file)
        return

    detector = BombDetector(
        model_dir=args.model_dir,
        input_dir=args.input_dir,
        output_dir=str(output_dir)
    )

    all_results = []
    for fname in tqdm(batch_list, desc="Files", unit="file"):
        all_results.extend(detector.run_inference(files=[fname]))

    # Save per-batch CSV
    df = pd.DataFrame(all_results, columns=["File", "Timestamp (HH:MM:SS)"])
    csv_name = Path(args.batch_file).stem + "_results.csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d detections to %s", len(df), csv_path)


if __name__ == "__main__":
    main()
