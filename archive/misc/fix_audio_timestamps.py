#!/usr/bin/env python3
"""
Shift UTC-timestamped WAV filenames to UTC+TIME_SHIFT_HOURS.

Looks for files named YYYYMMDD_HHMMSS.wav in FOLDER_PATH,
adds 8 hours to the timestamp (rolling date if needed),
and renames, avoiding overwrite by appending a counter.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FOLDER_PATH: str = "/media/bwilliams/226ECA8B6ECA56E7/four_islands_bombs/M36_pulau_pala"
TIME_SHIFT_HOURS: int = 8


def corrected_filename(filename: str) -> Optional[str]:
  """Return filename shifted by TIME_SHIFT_HOURS, or None if unparsable."""
  base, ext = os.path.splitext(filename)
  if ext.lower() != ".wav":
    return None

  try:
    date_str, time_str = base.split("_")
    original = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
  except (ValueError, IndexError):
    return None

  shifted = original + timedelta(hours=TIME_SHIFT_HOURS)
  return f"{shifted:%Y%m%d}_{shifted:%H%M%S}{ext}"


def main() -> None:
  """Rename all WAVs in FOLDER_PATH by shifting their embedded timestamps."""
  for fname in os.listdir(FOLDER_PATH):
    new_name = corrected_filename(fname)
    if not new_name or new_name == fname:
      if new_name is None:
        logging.warning(f"Skipping unparsed file: {fname}")
      continue

    old_path = os.path.join(FOLDER_PATH, fname)
    base, ext = os.path.splitext(new_name)
    unique, counter = new_name, 1

    # Avoid overwriting existing files
    while os.path.exists(os.path.join(FOLDER_PATH, unique)):
      unique = f"{base}_{counter}{ext}"
      counter += 1

    logging.info(f"Renaming {fname} → {unique}")
    os.rename(old_path, os.path.join(FOLDER_PATH, unique))


if __name__ == "__main__":
  main()
