#!/usr/bin/env python3
"""Count files starting with YB in processed_new_data directory."""

import os
import glob
from pathlib import Path


def count_yb_files():
    """Count all files starting with 'YB' in processed_new_data directory."""
    processed_dir = Path("data/processed_new_data")

    if not processed_dir.exists():
        print(f"Directory not found: {processed_dir}")
        return 0

    total_yb_files = 0
    month_counts = {}

    print("YB file counts by month:")
    print("=" * 40)

    # Process each month subdirectory
    for month_dir in sorted(processed_dir.iterdir()):
        if month_dir.is_dir():
            month_name = month_dir.name
            yb_files = list(month_dir.glob("YB*.wav"))
            yb_count = len(yb_files)

            month_counts[month_name] = yb_count
            total_yb_files += yb_count

            print(f"{month_name:20} : {yb_count:4d} YB files")

    print("=" * 40)
    print(f"{'TOTAL':20} : {total_yb_files:4d} YB files")

    return total_yb_files, month_counts


if __name__ == "__main__":
    count_yb_files()
