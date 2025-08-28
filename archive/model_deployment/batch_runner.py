#!/usr/bin/env python
"""
batch_runner.py

Split WAVs into batches, prompt once about overwrite, then invoke run_inference.py
for each batch (passing --output-dir to data/detections/<raw_name>). Also appends
all per-batch results into a single all_results.csv file.
"""

import os
import math
import shutil
import subprocess
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from config import MODEL_DIR, INPUT_DIR, DATA_DIR, BATCH_SIZE, SCRATCH_DIR, OUTPUT_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def main() -> None:
    # 1. Determine output dir like data/detections/<raw_name>
    raw_name = Path(INPUT_DIR).name
    detections_base = DATA_DIR / OUTPUT_FOLDER
    detections_base.mkdir(parents=True, exist_ok=True)
    output_dir = detections_base / raw_name

    # 2. Check if output dir exists and request user confirmation to overwrite
    if output_dir.exists():
        resp = input(
            f"Output directory '{output_dir}' already exists. "
            "Proceeding will clear ALL its contents. Continue? [Y/N]: "
        ).strip().lower()
        if resp != "y":
            print("Aborting.")
            sys.exit(0)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Set up scratch dir to store file lists for each batch
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Gather all input files
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")]
    if not files:
        logging.error(f"No WAV files found in {INPUT_DIR}")
        return

    total_batches = math.ceil(len(files) / BATCH_SIZE)
    logging.info("Processing %d files in %d batches (batch size=%d)",
                 len(files), total_batches, BATCH_SIZE)

    # 5. Prepare master CSV
    master_csv = output_dir / "all_results.csv"
    if master_csv.exists():
        master_csv.unlink()

    # 6. Loop over batches
    for batch_idx in tqdm(range(total_batches), desc="Batches", unit="batch"):
        start = batch_idx * BATCH_SIZE
        batch_files = files[start : start + BATCH_SIZE]

        batch_file = SCRATCH_DIR / f"batch_{batch_idx}.txt"
        batch_file.write_text("\n".join(batch_files))

        logging.info("Starting batch %d/%d with %d files…",
                     batch_idx+1, total_batches, len(batch_files))

        subprocess.run(
            [
                "python", "-m", "scripts.run_inference",
                "--model-dir", str(MODEL_DIR),
                "--input-dir", str(INPUT_DIR),
                "--output-dir", str(output_dir),
                "--batch-file", str(batch_file)
            ],
            check=True
        )

        # Append batch CSV to master CSV
        batch_csv = output_dir / f"batch_{batch_idx}_results.csv"
        if batch_csv.exists():
            df = pd.read_csv(batch_csv)
            df.to_csv(master_csv, mode="a", header=not master_csv.exists(), index=False)
            batch_csv.unlink()

        batch_file.unlink()
        logging.info("Finished batch %d/%d", batch_idx+1, total_batches)

    # 7. Sort the master CSV by File then Timestamp
    df = pd.read_csv(master_csv)
    df.sort_values(
        by=["File", "Timestamp (HH:MM:SS)"],
        inplace=True,
        ignore_index=True
    )
    df.to_csv(master_csv, index=False)

    # End 
    logging.info(f"All batches complete. Final results in: {master_csv}")


if __name__ == "__main__":
    main()
