#!/usr/bin/env python3
"""Count all bomb annotations across all CSV files."""

import pandas as pd
import glob
import os


def count_bombs_in_csvs():
    """Count all rows where Bombs = 'Y' across all CSV files."""
    spreadsheet_dir = "data/annotated_spreadsheets"

    total_bombs = 0
    file_counts = {}

    # Find all CSV files
    csv_pattern = os.path.join(spreadsheet_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    print(f"Found {len(csv_files)} CSV files:")
    print("=" * 50)

    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        try:
            df = pd.read_csv(csv_file)
            # Count rows where Bombs column equals 'Y'
            bomb_count = (df["Bombs"].astype(str) == "Y").sum()
            file_counts[filename] = bomb_count
            total_bombs += bomb_count
            print(f"{filename:30} : {bomb_count:4d} bombs")
        except Exception as e:
            print(f"{filename:30} : ERROR - {e}")
            file_counts[filename] = 0

    print("=" * 50)
    print(f"{'TOTAL':30} : {total_bombs:4d} bombs")
    return total_bombs, file_counts


if __name__ == "__main__":
    count_bombs_in_csvs()
