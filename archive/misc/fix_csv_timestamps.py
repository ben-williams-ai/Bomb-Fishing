#!/usr/bin/env python3
"""
Update a CSV's 'File' column by shifting UTC timestamps in filenames to 
UTC+TIME_SHIFT_HOURS.

Set CSV_PATH below and run this script to apply an 8-hour shift to each
'File' entry formatted as YYYYMMDD_HHMMSS.wav, overwriting the CSV.
"""
import csv
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

# Configuration
CSV_PATH: str = ""  
TIME_SHIFT_HOURS: int = 8

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def shift_filename(filename: str) -> Optional[str]:
    """
    Shift a single filename's timestamp by TIME_SHIFT_HOURS.
    Return new filename or None if unparsable.
    """
    if not filename.lower().endswith('.wav'):
        logging.warning(f"Skipping non-WAV entry: {filename}")
        return None

    base = filename[:-4]
    try:
        date_str, time_str = base.split('_')
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    except (ValueError, IndexError):
        logging.warning(f"Unparsable filename: {filename}")
        return None

    new_dt = dt + timedelta(hours=TIME_SHIFT_HOURS)
    return f"{new_dt:%Y%m%d}_{new_dt:%H%M%S}.wav"


def update_csv(csv_path: str) -> None:
    """
    Read CSV at csv_path, shift each 'File' field, and overwrite the file in place.
    """
    temp_path = f"{csv_path}.tmp"
    with open(csv_path, newline='', encoding='utf-8') as infile, \
         open(temp_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            orig = row.get('File', '')
            shifted = shift_filename(orig)
            if shifted:
                logging.info(f"CSV: {orig} → {shifted}")
                row['File'] = shifted
            writer.writerow(row)

    os.replace(temp_path, csv_path)
    logging.info(f"CSV updated: {csv_path}")


if __name__ == '__main__':
    update_csv(CSV_PATH)
