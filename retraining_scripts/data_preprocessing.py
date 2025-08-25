#!/usr/bin/env python3
"""
Data Extraction and Preprocessing
Bomb Detection Model Retraining
"""

import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import re
import subprocess
from typing import Tuple, Dict, Optional, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles the complete data preprocessing pipeline for bomb detection retraining."""

    def __init__(
        self,
        data_dir: str = "../data",
        input_compressed_dir: Optional[str] = None,
        input_annotations_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        extracted_dir: Optional[str] = None,
    ):
        """Initialize DataPreprocessor with configurable I/O directories.

        Args:
            data_dir: Base data directory (default: "../data")
            input_compressed_dir: Directory containing .tar.gz/.7z files (default: data_dir/compressed_new_data)
            input_annotations_dir: Directory containing .csv annotation files (default: data_dir/annotated_spreadsheets)
            output_dir: Output directory for processed files (default: data_dir/processed_new_data)
            extracted_dir: Directory containing already extracted files (default: data_dir/extracted)
        """
        self.data_dir = Path(data_dir)

        # Configure input directories
        self.compressed_dir = (
            Path(input_compressed_dir)
            if input_compressed_dir
            else self.data_dir / "compressed_new_data"
        )
        self.annotations_dir = (
            Path(input_annotations_dir)
            if input_annotations_dir
            else self.data_dir / "annotated_spreadsheets"
        )

        # Configure output directories
        self.extracted_dir = (
            Path(extracted_dir) if extracted_dir else self.data_dir / "extracted"
        )
        self.processed_dir = (
            Path(output_dir) if output_dir else self.data_dir / "processed_new_data"
        )
        self.final_dir = self.data_dir / "final_new_dataset"

        # Audio specifications
        self.target_sample_rate = 8000  # Hz (inference requirement)
        self.window_length = 2.88  # seconds
        self.window_samples = int(
            self.window_length * self.target_sample_rate
        )  # 23040 samples

        # Extracted files are already at 8 kHz (not 22.05 kHz as originally assumed)
        self.extracted_sample_rate = 8000  # Hz

        # Create directories
        for dir_path in [self.extracted_dir, self.processed_dir, self.final_dir]:
            dir_path.mkdir(exist_ok=True)

    @staticmethod
    def _format_seconds_for_filename(seconds_value: float) -> str:
        """Format a seconds float for inclusion in a filename.

        Returns a compact, filesystem-safe string like "34.0s" or "136.5s".
        """
        # Keep one decimal place for clarity when spreadsheets include tenths
        return f"{seconds_value:.1f}s"

    @staticmethod
    def _parse_h_mm_ss_timestamp_to_seconds(timestamp_token: str) -> Optional[float]:
        """Parse a token like '0.00.46' (H.MM.SS) into seconds (float)."""
        try:
            parts = timestamp_token.split(".")
            if len(parts) != 3:
                return None
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return float(hours * 3600 + minutes * 60 + seconds)
        except Exception:
            return None

    def extract_all_compressed_files(self) -> Dict[str, Path]:
        """Extract all compressed detection files and return mapping of month to directory."""
        logger.info("Step 1: Extracting all compressed detection files...")

        compressed_files = (
            list(self.compressed_dir.glob("*.tar.gz"))
            + list(self.compressed_dir.glob("*.7z"))
            + list(self.compressed_dir.glob("*.zip"))
        )
        month_dirs = {}

        for compressed_file in compressed_files:
            # Extract full base name without extension for exact CSV matching
            # e.g., "north_2023_jun_28.zip" -> "north_2023_jun_28"
            base_name = compressed_file.stem  # removes .zip/.7z extension

            # Extract just the date part for directory naming
            month_match = re.search(r"(\d{4}_\w+_\d+)", base_name)
            if not month_match:
                logger.warning(f"Could not parse month from {compressed_file.name}")
                continue

            month = month_match.group(1)  # Just the date part for directory
            month_dir = self.extracted_dir / month

            # Store the full base name for CSV lookup
            if not hasattr(self, "file_mappings"):
                self.file_mappings = {}
            self.file_mappings[month] = base_name

            if month_dir.exists():
                logger.info(f"Month {month} already extracted, skipping...")
                month_dirs[month] = month_dir
                continue

            logger.info(f"Extracting {compressed_file.name} to {month_dir}")

            try:
                # Extract using 7z
                if compressed_file.suffix == ".7z":
                    cmd = ["7z", "x", str(compressed_file), f"-o{month_dir}", "-y"]
                else:  # .zip
                    cmd = ["unzip", "-q", str(compressed_file), "-d", str(month_dir)]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )

                if result.returncode == 0:
                    logger.info(f"Successfully extracted {month}")
                    month_dirs[month] = month_dir
                else:
                    logger.error(f"Failed to extract {month}: {result.stderr}")

            except Exception as e:
                logger.error(f"Error extracting {month}: {e}")

        logger.info(f"Extracted {len(month_dirs)} months")
        return month_dirs

    def parse_csv_annotations(self, month: str) -> pd.DataFrame:
        """Parse CSV annotations for a given month using exact filename matching."""
        # Use the stored mapping to find the exact base name
        if hasattr(self, "file_mappings") and month in self.file_mappings:
            base_name = self.file_mappings[month]
            csv_file = self.annotations_dir / f"{base_name}.csv"
        else:
            # Fallback to old behavior for backwards compatibility
            month_parts = month.split("_")
            if len(month_parts) == 3:
                csv_month = f"{month_parts[0]}_{month_parts[1]}{month_parts[2]}"
            else:
                csv_month = month
            csv_file = self.annotations_dir / f"south_{csv_month}.csv"

        if not csv_file.exists():
            logger.warning(f"CSV file not found for {month}: {csv_file}")
            return pd.DataFrame()

        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} annotations for {month}")
        return df

    def extract_window_from_audio(
        self, audio_file: Path, start_time: float, duration: float = 2.88
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Extract a 2.88s window from detection files by cutting the first 1 second.

        Detection files are clips from the inference pipeline with different strategies:
        - 4s files: 1s pre-buffer + 2.88s detection + 0.12s post-buffer
        - 5s files: 1s pre-buffer + 2.88s detection + 1.12s post-buffer

        Strategy: Skip the first 1s (pre-buffer) and take the next 2.88s.
        This gives us the actual detection window that the model flagged.
        The start_time parameter is ignored since we always cut from 1s mark.
        """
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=None)

            # Calculate samples
            skip_samples = int(1.0 * sr)  # Skip first 1 second (pre-buffer)
            window_samples = int(duration * sr)  # 2.88 seconds

            # Check if file has enough audio after skipping 1s
            if len(audio) < skip_samples + window_samples:
                logger.warning(
                    f"File {audio_file.name} too short after skipping 1s pre-buffer "
                    f"({len(audio)/sr:.2f}s total, need {(skip_samples + window_samples)/sr:.2f}s)"
                )
                return None

            # Extract window: skip first 1s, take next 2.88s
            start_sample = skip_samples
            end_sample = start_sample + window_samples
            window = audio[start_sample:end_sample]

            # Ensure exact length (should already be correct, but safety check)
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)), "constant")
                logger.debug(f"Padded window for {audio_file.name}")
            elif len(window) > window_samples:
                window = window[:window_samples]
                logger.debug(f"Truncated window for {audio_file.name}")

            logger.debug(
                f"Extracted detection window from {audio_file.name}: "
                f"skipped 1s pre-buffer, took {duration}s detection window"
            )

            return window, sr

        except Exception as e:
            logger.error(f"Error extracting window from {audio_file}: {e}")
            return None

    def resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == self.target_sample_rate:
            return audio

        # Resample to target sample rate
        resampled = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=self.target_sample_rate
        )

        # Ensure exact length for target sample rate
        target_length = self.window_samples  # This is already calculated for 8kHz
        if len(resampled) > target_length:
            resampled = resampled[:target_length]
        elif len(resampled) < target_length:
            resampled = np.pad(
                resampled, (0, target_length - len(resampled)), "constant"
            )

        return resampled

    def process_month_data(self, month: str, month_dir: Path) -> Tuple[int, int]:
        """Process all detection files for a given month."""
        logger.info(f"Processing month: {month}")

        # Parse annotations
        annotations_df = self.parse_csv_annotations(month)
        if annotations_df.empty:
            logger.warning(f"No annotations found for {month}")
            return 0, 0

        # Find detection files
        # (robust to an extra nested month folder which happens when zipping on linux)
        detection_dir = month_dir / "Detected_bombs"
        if not detection_dir.exists():
            # Fallback: allow exactly one extra level, e.g., month_dir/<month>/Detected_bombs
            candidates = list(month_dir.glob("*/Detected_bombs"))
            if candidates:
                detection_dir = candidates[0]
                logger.info(f"Using nested detection directory: {detection_dir}")
            else:
                logger.warning(f"Detection directory not found: {month_dir}/Detected_bombs (or nested)")
                return 0, 0

        detection_files = list(detection_dir.glob("*.wav"))
        logger.info(f"Found {len(detection_files)} detection files for {month}")

        # Create output directories
        month_processed_dir = self.processed_dir / month
        month_processed_dir.mkdir(exist_ok=True)

        yb_count = 0
        nb_count = 0

        # Process each detection file
        for detection_file in detection_files:
            # Parse filename to get timestamp
            # Format: 0000003_0.00.46_20231107_014403.wav
            filename_match = re.search(
                r"(\d+)_(\d+\.\d+\.\d+)_(\d{8}_\d{6})", detection_file.name
            )
            if not filename_match:
                logger.warning(f"Could not parse filename: {detection_file.name}")
                continue

            file_id = filename_match.group(1)
            timestamp_str = filename_match.group(2)
            datetime_str = filename_match.group(3)

            # Convert timestamp to seconds
            # Format: "0.00.34" = 0 hours, 00 minutes, 34 seconds = 34.0 seconds
            try:
                time_parts = timestamp_str.split(".")
                if len(time_parts) == 3:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = int(time_parts[2])
                    # Convert to total seconds: 0.00.34 = 0*3600 + 0*60 + 34 = 34s
                    timestamp_seconds = hours * 3600 + minutes * 60 + seconds
                else:
                    logger.warning(f"Invalid timestamp format: {timestamp_str}")
                    continue
            except ValueError:
                logger.warning(f"Could not parse timestamp: {timestamp_str}")
                continue

            # Check if this timestamp has a confirmed bomb
            # FIXED: Now matches specific timestamp, not just any bomb in the file

            # Skip detection files if no annotations are available for this month
            if annotations_df.empty:
                logger.info(
                    f"Skipping detection {detection_file.name}: "
                    f"no annotations available for month {month}"
                )
                continue

            is_bomb = False

            # Get all annotations for this file
            file_annotations = annotations_df[
                annotations_df["File"] == f"{datetime_str}.WAV"
            ]

            # Skip detection files that have no corresponding annotations
            if file_annotations.empty:
                logger.info(
                    f"Skipping detection {detection_file.name}: "
                    f"no annotations found for {datetime_str}.WAV"
                )
                continue

            # Convert annotation timestamps to seconds for comparison
            for _, annotation in file_annotations.iterrows():
                if annotation["Bombs"] == "Y":
                    try:
                        # Parse annotation timestamp (formats: "00:25.4", "0:00:36")
                        timestamp_str_ann = str(annotation["Timestamp"])

                        # Handle different timestamp formats
                        if ":" in timestamp_str_ann:
                            # Format: "00:25.4" or "0:00:36"
                            if timestamp_str_ann.count(":") == 1:
                                # MM:SS.T format
                                parts = timestamp_str_ann.split(":")
                                minutes = int(parts[0])
                                seconds_part = parts[1]
                                if "." in seconds_part:
                                    sec_parts = seconds_part.split(".")
                                    seconds = int(sec_parts[0])
                                    tenths = (
                                        int(sec_parts[1]) if len(sec_parts) > 1 else 0
                                    )
                                else:
                                    seconds = int(seconds_part)
                                    tenths = 0
                                annotation_seconds = (
                                    minutes * 60 + seconds + tenths / 10
                                )
                            else:
                                # H:MM:SS format
                                parts = timestamp_str_ann.split(":")
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = int(parts[2])
                                annotation_seconds = (
                                    hours * 3600 + minutes * 60 + seconds
                                )
                        else:
                            # Skip malformed timestamps
                            continue

                        # Check if timestamps match within tolerance (±2 seconds)
                        tolerance = 2.0
                        if abs(timestamp_seconds - annotation_seconds) <= tolerance:
                            is_bomb = True
                            logger.info(
                                f"Bomb match: detection={timestamp_seconds:.1f}s, "
                                f"annotation={annotation_seconds:.1f}s"
                            )
                            break

                    except (ValueError, AttributeError) as e:
                        logger.warning(
                            f"Could not parse annotation timestamp '{annotation['Timestamp']}': {e}"
                        )
                        continue

            # Extract window
            window_result = self.extract_window_from_audio(
                detection_file, timestamp_seconds
            )
            if window_result is None:
                continue

            window_data, actual_sample_rate = window_result

            # Resample (this will be a no-op since we're already at 8 kHz)
            resampled = self.resample_audio(window_data, actual_sample_rate)

            # Save with appropriate naming
            if is_bomb:
                prefix = "YB"
                yb_count += 1
            else:
                prefix = "NB"
                nb_count += 1

            # Create filename similar to original format, but include detection timestamp
            det_token = self._format_seconds_for_filename(timestamp_seconds)

            # If matched to an annotation timestamp, include it too for traceability
            ann_token: Optional[str] = None
            if is_bomb:
                # Recompute the matched annotation seconds to embed (best-effort):
                # We already looped annotations; repeat minimal conversion to capture the first match
                file_annotations = annotations_df[
                    annotations_df["File"] == f"{datetime_str}.WAV"
                ]
                if not file_annotations.empty:
                    tolerance = 2.0
                    for _, annotation in file_annotations.iterrows():
                        if annotation.get("Bombs") == "Y":
                            try:
                                timestamp_str_ann = str(annotation["Timestamp"])
                                if ":" in timestamp_str_ann:
                                    if timestamp_str_ann.count(":") == 1:
                                        parts = timestamp_str_ann.split(":")
                                        minutes = int(parts[0])
                                        seconds_part = parts[1]
                                        if "." in seconds_part:
                                            sec_parts = seconds_part.split(".")
                                            seconds = int(sec_parts[0])
                                            tenths = (
                                                int(sec_parts[1])
                                                if len(sec_parts) > 1
                                                else 0
                                            )
                                        else:
                                            seconds = int(seconds_part)
                                            tenths = 0
                                        candidate = (
                                            minutes * 60 + seconds + tenths / 10.0
                                        )
                                    else:
                                        parts = timestamp_str_ann.split(":")
                                        hours = int(parts[0])
                                        minutes = int(parts[1])
                                        seconds = int(parts[2])
                                        candidate = (
                                            hours * 3600 + minutes * 60 + seconds
                                        )
                                else:
                                    continue

                                if abs(timestamp_seconds - candidate) <= tolerance:
                                    ann_token = self._format_seconds_for_filename(
                                        candidate
                                    )
                                    break
                            except Exception:
                                continue

            if ann_token:
                output_filename = f"{prefix}{int(file_id):06d}_M01_{datetime_str}_det{det_token}_ann{ann_token}.wav"
            else:
                output_filename = (
                    f"{prefix}{int(file_id):06d}_M01_{datetime_str}_det{det_token}.wav"
                )
            output_path = month_processed_dir / output_filename

            # Save audio
            sf.write(output_path, resampled, self.target_sample_rate)

        logger.info(f"Processed {month}: {yb_count} bombs, {nb_count} non-bombs")
        return yb_count, nb_count

    def apply_augmentation(self):
        """Apply augmentation to the processed data."""
        logger.info("Step 5: Applying augmentation...")

        # This would use the existing augmentation code from archive
        # For now, we'll create a placeholder
        logger.info("Augmentation will be applied in the next step using existing code")

    def create_final_dataset(self):
        """Create the final dataset structure."""
        logger.info("Creating final dataset structure...")

        # Note: Train/test split is handled by separate create_train_test_split.py script
        # This method is kept for compatibility but doesn't create directories
        logger.info(
            "Final dataset structure will be created by train/test split script"
        )

    def get_existing_extracted_dirs(self) -> Dict[str, Path]:
        """Get mapping of already extracted month directories and create file mappings."""
        logger.info("Looking for existing extracted directories...")

        month_dirs = {}
        if not self.extracted_dir.exists():
            logger.error(f"Extracted directory not found: {self.extracted_dir}")
            return month_dirs

        # Initialize file mappings
        if not hasattr(self, "file_mappings"):
            self.file_mappings = {}

        for month_dir in self.extracted_dir.iterdir():
            if month_dir.is_dir():
                month = month_dir.name
                if (month_dir / "Detected_bombs").exists():
                    month_dirs[month] = month_dir
                    logger.info(f"Found extracted month: {month}")

                    # Try to find corresponding CSV file to determine the correct prefix
                    self._find_csv_mapping_for_month(month)
                else:
                    logger.warning(f"Month directory {month} missing Detected_bombs subdirectory")

        logger.info(f"Found {len(month_dirs)} extracted months")
        return month_dirs

    def _find_csv_mapping_for_month(self, month: str):
        """Find the correct CSV file for a month by trying different prefixes."""
        # Try to find a matching CSV file with any prefix
        csv_files = list(self.annotations_dir.glob(f"*{month}*.csv"))

        if csv_files:
            # Use the first matching CSV file's base name
            csv_base = csv_files[0].stem  # Remove .csv extension
            self.file_mappings[month] = csv_base
            logger.info(f"Mapped {month} to CSV file: {csv_base}.csv")
        else:
            # Try alternative naming pattern (YYYY_MMMdd format)
            month_parts = month.split("_")
            if len(month_parts) == 3:
                alt_month = f"{month_parts[0]}_{month_parts[1]}{month_parts[2]}"
                alt_csv_files = list(self.annotations_dir.glob(f"*{alt_month}*.csv"))
                if alt_csv_files:
                    csv_base = alt_csv_files[0].stem
                    self.file_mappings[month] = csv_base
                    logger.info(f"Mapped {month} to CSV file: {csv_base}.csv")

    def run_complete_pipeline(self, skip_extraction: bool = False):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting Data Extraction and Preprocessing")

        if skip_extraction:
            # Use already extracted files
            month_dirs = self.get_existing_extracted_dirs()
        else:
            # Step 1: Extract all compressed files
            month_dirs = self.extract_all_compressed_files()

        # Step 2-4: Process each month
        total_yb = 0
        total_nb = 0

        for month, month_dir in month_dirs.items():
            yb_count, nb_count = self.process_month_data(month, month_dir)
            total_yb += yb_count
            total_nb += nb_count

        logger.info(f"Processing complete: {total_yb} bombs, {total_nb} non-bombs")

        # Step 5: Apply augmentation (placeholder)
        self.apply_augmentation()

        # Create final dataset
        self.create_final_dataset()

        logger.info("Preprocessing complete!")

    # ----------------------
    # Debug/verification API
    # ----------------------
    def verify_files_against_annotations(
        self, month: str, processed_filenames: List[str], tolerance_seconds: float = 2.0
    ) -> List[Dict[str, object]]:
        """Verify a set of processed filenames against the spreadsheet annotations.

        Returns a list of dicts with fields:
          - filename
          - label (YB/NB)
          - datetime_str
          - detection_seconds
          - matched_annotation_seconds (or None)
          - spreadsheet_has_bomb_near_detection (bool)
          - conclusion (str)
        """
        results: List[Dict[str, object]] = []

        # Ensure detection directory exists
        detection_dir = self.extracted_dir / month / "Detected_bombs"
        if not detection_dir.exists():
            logger.error(f"Detection directory not found: {detection_dir}")
            return results

        # Load annotations for month
        annotations_df = self.parse_csv_annotations(month)
        if annotations_df.empty:
            logger.error(f"No annotations available for month {month}")
            return results

        # Index detection files by numeric id and datetime suffix for quick lookup
        id_datetime_to_path: Dict[Tuple[int, str], Path] = {}
        for wav_path in detection_dir.glob("*.wav"):
            name = wav_path.name
            m = re.match(
                r"^(\d+)_([0-9][0-9]?\.[0-9]{2}\.[0-9]{2})_(\d{8}_\d{6})\\.wav$", name
            )
            if not m:
                continue
            det_id_int = int(m.group(1))
            det_datetime = m.group(3)
            id_datetime_to_path[(det_id_int, det_datetime)] = wav_path

        for proc_name in processed_filenames:
            base = proc_name
            if base.endswith(".wav"):
                base = base[:-4]

            m = re.match(r"^(YB|NB)(\d{6})_M\d{2}_(\d{8}_\d{6})", base)
            if not m:
                logger.warning(f"Unrecognized processed filename format: {proc_name}")
                continue
            label = m.group(1)
            proc_id_int = int(m.group(2))
            datetime_str = m.group(3)

            # Try to find the corresponding detection file using numeric id + datetime
            det_path = id_datetime_to_path.get((proc_id_int, datetime_str))
            detection_seconds: Optional[float] = None
            if det_path is not None:
                # Parse detection timestamp seconds from detection filename
                det_match = re.match(
                    r"^(\d+)_([0-9][0-9]?\.[0-9]{2}\.[0-9]{2})_(\d{8}_\d{6})\\.wav$",
                    det_path.name,
                )
                if not det_match:
                    logger.warning(
                        f"Could not parse detection filename: {det_path.name}"
                    )
                else:
                    det_ts_token = det_match.group(2)
                    detection_seconds = self._parse_h_mm_ss_timestamp_to_seconds(
                        det_ts_token
                    )
                    if detection_seconds is None:
                        logger.warning(
                            f"Could not parse seconds from token: {det_ts_token}"
                        )

            # Look up annotations for the underlying .WAV
            file_annotations = annotations_df[
                annotations_df["File"] == f"{datetime_str}.WAV"
            ]

            matched_annotation_seconds: Optional[float] = None
            spreadsheet_has_bomb_near_detection = False
            spreadsheet_has_any_bomb_in_file = False
            if not file_annotations.empty:
                # File-level presence of any Y annotations
                spreadsheet_has_any_bomb_in_file = (
                    file_annotations["Bombs"].astype(str) == "Y"
                ).any()
                for _, annotation in file_annotations.iterrows():
                    if str(annotation.get("Bombs")) != "Y":
                        continue
                    try:
                        timestamp_str_ann = str(annotation["Timestamp"])
                        if ":" not in timestamp_str_ann:
                            continue
                        if timestamp_str_ann.count(":") == 1:
                            parts = timestamp_str_ann.split(":")
                            minutes = int(parts[0])
                            seconds_part = parts[1]
                            if "." in seconds_part:
                                sec_parts = seconds_part.split(".")
                                seconds = int(sec_parts[0])
                                tenths = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                            else:
                                seconds = int(seconds_part)
                                tenths = 0
                            ann_seconds = minutes * 60 + seconds + tenths / 10.0
                        else:
                            parts = timestamp_str_ann.split(":")
                            hours = int(parts[0])
                            minutes = int(parts[1])
                            seconds = int(parts[2])
                            ann_seconds = hours * 3600 + minutes * 60 + seconds

                        # Only compute proximity match if we have detection_seconds
                        if (
                            detection_seconds is not None
                            and abs(detection_seconds - ann_seconds)
                            <= tolerance_seconds
                        ):
                            spreadsheet_has_bomb_near_detection = True
                            matched_annotation_seconds = ann_seconds
                            break
                    except Exception:
                        continue

            # Build conclusion
            if detection_seconds is None:
                # Fallback to file-level verdict only
                if label == "YB" and not spreadsheet_has_any_bomb_in_file:
                    conclusion = "YB but spreadsheet has no Y in file (possible FP or spreadsheet error)"
                elif label == "NB" and spreadsheet_has_any_bomb_in_file:
                    conclusion = "NB but spreadsheet has at least one Y in file (possible FN or code issue)"
                else:
                    conclusion = "Consistent at file level"
            else:
                if label == "YB" and not spreadsheet_has_bomb_near_detection:
                    conclusion = "YB but spreadsheet has no Y near detection (possible FP or spreadsheet error)"
                elif label == "NB" and spreadsheet_has_bomb_near_detection:
                    conclusion = "NB but spreadsheet has Y near detection (possible FN or code issue)"
                else:
                    conclusion = "Consistent with spreadsheet"

            results.append(
                {
                    "filename": proc_name,
                    "label": label,
                    "datetime_str": datetime_str,
                    "detection_seconds": detection_seconds,
                    "matched_annotation_seconds": matched_annotation_seconds,
                    "spreadsheet_has_bomb_near_detection": spreadsheet_has_bomb_near_detection,
                    "spreadsheet_has_any_bomb_in_file": spreadsheet_has_any_bomb_in_file,
                    "conclusion": conclusion,
                }
            )

        return results


def main():
    """Run the data preprocessing pipeline or verification utility based on CLI args."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data preprocessing and verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O directory arguments
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Base data directory"
    )
    parser.add_argument(
        "--input-compressed-dir",
        type=str,
        help="Directory containing compressed data files (.tar.gz/.7z files). Default: data-dir/compressed_new_data",
    )
    parser.add_argument(
        "--input-annotations-dir",
        type=str,
        help="Directory containing CSV annotation files. Default: data-dir/annotated_spreadsheets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for processed files. Default: data-dir/processed_new_data",
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        help="Directory containing already extracted files. Default: data-dir/extracted",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction step and use already extracted files",
    )

    # Verification arguments
    parser.add_argument(
        "--verify-month",
        type=str,
        help="Month directory name (e.g., 2024_mar_12) for verification",
    )
    parser.add_argument(
        "--verify-files",
        type=str,
        nargs="*",
        help="Processed filenames to verify (e.g., YB000361_M01_20231123_135400.wav)",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=2.0,
        help="Tolerance (seconds) when matching detection to annotation timestamps",
    )

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        input_compressed_dir=args.input_compressed_dir,
        input_annotations_dir=args.input_annotations_dir,
        output_dir=args.output_dir,
        extracted_dir=args.extracted_dir,
    )

    if args.verify_month and args.verify_files:
        results = preprocessor.verify_files_against_annotations(
            args.verify_month,
            args.verify_files,
            tolerance_seconds=args.tolerance_seconds,
        )
        # Print a compact report
        for r in results:
            det_seconds = r.get("detection_seconds")
            ann_seconds = r.get("matched_annotation_seconds")
            det_tok = f"{det_seconds:.1f}s" if det_seconds is not None else "?"
            ann_tok = f"{ann_seconds:.1f}s" if ann_seconds is not None else "-"
            print(
                (
                    f"{r['filename']}: label={r['label']}, file={r['datetime_str']}, "
                    f"det={det_tok}, ann={ann_tok}, match={r['spreadsheet_has_bomb_near_detection']}, "
                    f"{r['conclusion']}"
                )
            )
    else:
        preprocessor.run_complete_pipeline(skip_extraction=args.skip_extraction)


if __name__ == "__main__":
    main()
