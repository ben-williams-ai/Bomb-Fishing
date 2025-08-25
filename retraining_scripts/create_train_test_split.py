#!/usr/bin/env python3
"""
Create configurable train/test split for processed audio data
Supports custom month assignments and flexible input/output directories
"""

import shutil
from pathlib import Path
import logging
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainTestSplitter:

    def __init__(
        self,
        data_dir: str = "../data",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        train_months: Optional[list] = None,
        test_months: Optional[list] = None,
    ):
        """Initialize TrainTestSplitter with configurable I/O directories.

        Args:
            data_dir: Base data directory (default: "../data")
            input_dir: Directory containing processed audio files (default: data_dir/processed_new_data)
            output_dir: Output directory for train/test split (default: data_dir/final_new_dataset)
            train_months: List of months to include in training set (default: None, uses default split)
            test_months: List of months to include in test set (default: None, uses default split)
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = (
            Path(input_dir) if input_dir else self.data_dir / "processed_new_data"
        )
        self.final_dir = (
            Path(output_dir) if output_dir else self.data_dir / "final_new_dataset"
        )

        # Store custom month assignments
        self.custom_train_months = train_months
        self.custom_test_months = test_months

        # Create output directories
        self.train_dir = self.final_dir / "train"
        self.test_dir = self.final_dir / "test"

        for dir_path in [self.train_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_available_months(self):
        """Get list of available months in processed data."""
        if not self.processed_dir.exists():
            logger.error(f"Processed data directory not found: {self.processed_dir}")
            return []

        months = [d.name for d in self.processed_dir.iterdir() if d.is_dir()]
        logger.info(f"Available months: {months}")
        return months

    def count_files_by_type(self, month_dir: Path):
        """Count YB and NB files in a month directory."""
        yb_count = len(list(month_dir.glob("YB*.wav")))
        nb_count = len(list(month_dir.glob("NB*.wav")))
        return yb_count, nb_count

    def analyze_data_distribution(self):
        """Analyze the distribution of data across months."""
        logger.info("=== DATA DISTRIBUTION ANALYSIS ===")

        available_months = self.get_available_months()
        total_yb = 0
        total_nb = 0

        for month in available_months:
            month_dir = self.processed_dir / month
            yb_count, nb_count = self.count_files_by_type(month_dir)
            total_yb += yb_count
            total_nb += nb_count

            logger.info(
                f"{month}: {yb_count} bombs, {nb_count} non-bombs ({yb_count + nb_count} total)"
            )

        logger.info(
            f"TOTAL: {total_yb} bombs, {total_nb} non-bombs ({total_yb + total_nb} total)"
        )

        # Determine train/test split
        if self.custom_train_months is not None and self.custom_test_months is not None:
            # Use custom month assignments
            train_months = [
                m for m in self.custom_train_months if m in available_months
            ]
            test_months = [m for m in self.custom_test_months if m in available_months]
            logger.info("Using custom month assignments")
        else:
            # Default: all months go to test split (for evaluation purposes)
            train_months = []  # No training data by default
            test_months = available_months  # All available months go to test
            logger.info("Using default split: all months in test set")

        logger.info(f"\nTrain months: {train_months}")
        logger.info(f"Test months: {test_months}")

        return train_months, test_months

    def copy_files_to_split(self, source_dir: Path, target_dir: Path, file_type: str):
        """Copy files of a specific type to the target directory."""
        files = list(source_dir.glob(f"{file_type}*.wav"))
        for file in files:
            target_file = target_dir / file.name
            shutil.copy2(file, target_file)
        return len(files)

    def create_split(self):
        """Create the train/test split."""
        logger.info("=== CREATING TRAIN/TEST SPLIT ===")

        # Analyze current data distribution
        train_months, test_months = self.analyze_data_distribution()

        if not train_months and not test_months:
            logger.error("No months available for processing!")
            return

        # Clear existing directories
        for split_dir in [self.train_dir, self.test_dir]:
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

        # Copy training data
        train_yb_total = 0
        train_nb_total = 0
        if train_months:
            logger.info(
                f"Copying data from {len(train_months)} months to train split..."
            )
            for month in train_months:
                month_dir = self.processed_dir / month
                if month_dir.exists():
                    yb_count = self.copy_files_to_split(month_dir, self.train_dir, "YB")
                    nb_count = self.copy_files_to_split(month_dir, self.train_dir, "NB")
                    train_yb_total += yb_count
                    train_nb_total += nb_count
                    logger.info(
                        f"  TRAIN {month}: {yb_count} bombs, {nb_count} non-bombs"
                    )

        # Copy test data
        test_yb_total = 0
        test_nb_total = 0
        if test_months:
            logger.info(f"Copying data from {len(test_months)} months to test split...")
            for month in test_months:
                month_dir = self.processed_dir / month
                if month_dir.exists():
                    yb_count = self.copy_files_to_split(month_dir, self.test_dir, "YB")
                    nb_count = self.copy_files_to_split(month_dir, self.test_dir, "NB")
                    test_yb_total += yb_count
                    test_nb_total += nb_count
                    logger.info(
                        f"  TEST {month}: {yb_count} bombs, {nb_count} non-bombs"
                    )

        # Summary
        logger.info("\n=== SPLIT SUMMARY ===")
        logger.info(
            f"Train: {train_yb_total} bombs, {train_nb_total} non-bombs ({train_yb_total + train_nb_total} total)"
        )
        logger.info(
            f"Test:  {test_yb_total} bombs, {test_nb_total} non-bombs ({test_yb_total + test_nb_total} total)"
        )
        logger.info(f"Total: {test_yb_total} bombs, {test_nb_total} non-bombs")

        # Calculate split ratio
        total_files = train_yb_total + train_nb_total + test_yb_total + test_nb_total
        if total_files > 0:
            train_ratio = (train_yb_total + train_nb_total) / total_files * 100
            test_ratio = (test_yb_total + test_nb_total) / total_files * 100
            logger.info(
                f"Split ratio: {train_ratio:.1f}% train, {test_ratio:.1f}% test"
            )
        else:
            logger.info("Split ratio: 0% train, 0% test (no files processed)")

        # Verify file counts
        train_files = len(list(self.train_dir.glob("*.wav")))
        test_files = len(list(self.test_dir.glob("*.wav")))

        logger.info(f"Verification: {train_files} train files, {test_files} test files")

        return {
            "train_months": train_months,
            "test_months": test_months,
            "train_yb": train_yb_total,
            "train_nb": train_nb_total,
            "test_yb": test_yb_total,
            "test_nb": test_nb_total,
        }

    def create_metadata_file(self, split_info: dict):
        """Create a metadata file documenting the split."""
        metadata_file = self.final_dir / "split_metadata.txt"

        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write("TRAIN/TEST SPLIT METADATA\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Created: {split_info.get('timestamp', 'N/A')}\n\n")
            f.write("SPLIT CONFIGURATION:\n")
            f.write(f"  Train months: {split_info['train_months']}\n")
            f.write(f"  Test months: {split_info['test_months']}\n\n")
            f.write("DATA COUNTS:\n")
            f.write(f"  Train bombs (YB): {split_info['train_yb']}\n")
            f.write(f"  Train non-bombs (NB): {split_info['train_nb']}\n")
            f.write(
                f"  Train total: {split_info['train_yb'] + split_info['train_nb']}\n\n"
            )
            f.write(f"  Test bombs (YB): {split_info['test_yb']}\n")
            f.write(f"  Test non-bombs (NB): {split_info['test_nb']}\n")
            f.write(
                f"  Test total: {split_info['test_yb'] + split_info['test_nb']}\n\n"
            )
            f.write("NOTES:\n")
            f.write("  - Configurable train/test split for processed audio data\n")
            f.write("  - Supports custom month assignments via command line\n")
            f.write("  - Default behavior: all months in test split\n")
            f.write("  - Audio files are YB (bomb) and NB (non-bomb) WAV files\n")

        logger.info(f"Metadata saved to: {metadata_file}")


def main():
    """Run the train/test split creation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train/test split from processed audio data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O directory arguments
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Base data directory"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing processed audio files. Default: data-dir/processed_new_data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for train/test split. Default: data-dir/final_new_dataset",
    )
    parser.add_argument(
        "--train-months",
        type=str,
        nargs="*",
        help="Months to include in training set (e.g., 2023_jun_28 2023_nov_23)",
    )
    parser.add_argument(
        "--test-months",
        type=str,
        nargs="*",
        help="Months to include in test set (e.g., 2023_jun_28 2023_nov_23)",
    )

    args = parser.parse_args()

    splitter = TrainTestSplitter(
        data_dir=args.data_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_months=args.train_months,
        test_months=args.test_months,
    )
    split_info = splitter.create_split()

    if split_info:
        splitter.create_metadata_file(split_info)
        logger.info("Train/test split creation complete!")
    else:
        logger.error("Failed to create train/test split!")


if __name__ == "__main__":
    main()
