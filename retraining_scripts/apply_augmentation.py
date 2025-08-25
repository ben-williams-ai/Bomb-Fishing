#!/usr/bin/env python3
"""
Apply augmentation to handle data imbalance
Uses exact augmentation logic from augmentations.ipynb
Strategy: 10x for bombs (YB), 2x for non-bombs (NB)
"""

import librosa
import soundfile as sf
import os
import glob
from pathlib import Path
import logging
from typing import Optional
from audiomentations import (
    Compose,
    AddGaussianNoise,
    PitchShift,
    TimeStretch,
    ClippingDistortion,
    Gain,
    SevenBandParametricEQ,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioAugmenter:

    def __init__(
        self,
        data_dir: str = "../data",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize AudioAugmenter with configurable I/O directories.

        Args:
            data_dir: Base data directory (default: "../data")
            input_dir: Directory containing training audio files (default: data_dir/final_new_dataset/train)
            output_dir: Output directory for augmented files (default: data_dir/final_new_dataset/train_augmented)
        """
        self.data_dir = Path(data_dir)
        self.train_dir = (
            Path(input_dir)
            if input_dir
            else self.data_dir / "final_new_dataset" / "train"
        )
        self.augmented_dir = (
            Path(output_dir)
            if output_dir
            else self.data_dir / "final_new_dataset" / "train_augmented"
        )

        # Sample rate
        self.sample_rate = 8000

        # Augmentation strategy
        self.yb_multiplier = 10  # 10x for bombs
        self.nb_multiplier = 2  # 2x for non-bombs

        # Create output directory
        self.augmented_dir.mkdir(parents=True, exist_ok=True)

        # Raw audio augmentation - EXACT SAME AS NOTEBOOK
        self.augment_raw_audio = Compose(
            [
                AddGaussianNoise(
                    min_amplitude=0.0001, max_amplitude=0.0005, p=0.5
                ),  # good
                PitchShift(
                    min_semitones=-2, max_semitones=12, p=0.5
                ),  # set values so it doesnt shift too low, removing bomb signal
                TimeStretch(p=0.5),  # defaults are fine
                ClippingDistortion(0, 5, p=0.5),  # tested params to make sure its good
                Gain(-10, 5, p=0.5),  # defaults are fine
                SevenBandParametricEQ(-12, 12, p=0.5),
            ]
        )

    def get_file_counts(self):
        """Get counts of YB and NB files in training directory."""
        yb_files = list(self.train_dir.glob("YB*.wav"))
        nb_files = list(self.train_dir.glob("NB*.wav"))

        return len(yb_files), len(nb_files)

    def copy_original_files(self):
        """Copy all original files to augmented directory first."""
        logger.info("Copying original files to augmented directory...")

        # Copy all original files
        original_files = list(self.train_dir.glob("*.wav"))
        for file in original_files:
            target_file = self.augmented_dir / file.name
            sf.write(target_file, *librosa.load(file, sr=self.sample_rate))

        logger.info(f"Copied {len(original_files)} original files")

    def augment_file(self, file_path: Path, base_name: str, multiplier: int):
        """Augment a single file multiple times."""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)

            # Create multiple augmented versions
            for i in range(
                1, multiplier
            ):  # Start from 1 since we already have original
                # Apply augmentation
                augmented_signal = self.augment_raw_audio(audio, sr)

                # Create output filename
                output_filename = f"{base_name}_aug{i}.wav"
                output_path = self.augmented_dir / output_filename

                # Save augmented file
                sf.write(output_path, augmented_signal, sr)

            return multiplier - 1  # Return number of new files created

        except Exception as e:
            logger.error(f"Error augmenting {file_path}: {e}")
            return 0

    def apply_augmentation(self):
        """Apply augmentation strategy to handle data imbalance."""
        logger.info("=== APPLYING AUGMENTATION STRATEGY ===")

        # Get current file counts
        yb_count, nb_count = self.get_file_counts()
        logger.info(f"Current counts: {yb_count} bombs, {nb_count} non-bombs")

        # Copy original files first
        self.copy_original_files()

        # Augment YB files (bombs) - 10x
        logger.info(f"\nAugmenting {yb_count} bomb files (10x each)...")
        yb_files = list(self.train_dir.glob("YB*.wav"))
        yb_augmented = 0

        for i, file_path in enumerate(yb_files):
            if i % 100 == 0:
                logger.info(f"  Processing bomb file {i+1}/{len(yb_files)}")

            base_name = file_path.stem  # Remove .wav extension
            new_files = self.augment_file(file_path, base_name, self.yb_multiplier)
            yb_augmented += new_files

        # Augment NB files (non-bombs) - 2x
        logger.info(f"\nAugmenting {nb_count} non-bomb files (2x each)...")
        nb_files = list(self.train_dir.glob("NB*.wav"))
        nb_augmented = 0

        for i, file_path in enumerate(nb_files):
            if i % 1000 == 0:
                logger.info(f"  Processing non-bomb file {i+1}/{len(nb_files)}")

            base_name = file_path.stem  # Remove .wav extension
            new_files = self.augment_file(file_path, base_name, self.nb_multiplier)
            nb_augmented += new_files

        # Summary
        logger.info(f"\n=== AUGMENTATION SUMMARY ===")
        logger.info(f"Original bombs: {yb_count}")
        logger.info(f"Original non-bombs: {nb_count}")
        logger.info(f"New bomb files created: {yb_augmented}")
        logger.info(f"New non-bomb files created: {nb_augmented}")
        logger.info(
            f"Total files after augmentation: {yb_count + nb_count + yb_augmented + nb_augmented}"
        )

        # Calculate new ratios
        total_yb = yb_count + yb_augmented
        total_nb = nb_count + nb_augmented
        ratio = total_nb / total_yb if total_yb > 0 else float("inf")

        logger.info(f"New ratio (non-bombs:bombs): {ratio:.1f}:1")

        # Verify final counts
        final_yb = len(list(self.augmented_dir.glob("YB*.wav")))
        final_nb = len(list(self.augmented_dir.glob("NB*.wav")))
        logger.info(
            f"Verification - Final counts: {final_yb} bombs, {final_nb} non-bombs"
        )

        return {
            "original_yb": yb_count,
            "original_nb": nb_count,
            "new_yb": yb_augmented,
            "new_nb": nb_augmented,
            "final_yb": final_yb,
            "final_nb": final_nb,
            "ratio": ratio,
        }

    def create_metadata_file(self, aug_info: dict):
        """Create metadata file documenting the augmentation."""
        metadata_file = self.augmented_dir / "augmentation_metadata.txt"

        with open(metadata_file, "w") as f:
            f.write("AUGMENTATION METADATA\n")
            f.write("=" * 50 + "\n\n")
            f.write("AUGMENTATION STRATEGY:\n")
            f.write(f"  Bombs (YB): {self.yb_multiplier}x augmentation\n")
            f.write(f"  Non-bombs (NB): {self.nb_multiplier}x augmentation\n\n")
            f.write("AUGMENTATION PARAMETERS:\n")
            f.write(
                "  - AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.0005, p=0.5)\n"
            )
            f.write("  - PitchShift(min_semitones=-2, max_semitones=12, p=0.5)\n")
            f.write("  - TimeStretch(p=0.5)\n")
            f.write("  - ClippingDistortion(0, 5, p=0.5)\n")
            f.write("  - Gain(-10, 5, p=0.5)\n")
            f.write("  - SevenBandParametricEQ(-12, 12, p=0.5)\n\n")
            f.write("RESULTS:\n")
            f.write(f"  Original bombs: {aug_info['original_yb']}\n")
            f.write(f"  Original non-bombs: {aug_info['original_nb']}\n")
            f.write(f"  New bomb files created: {aug_info['new_yb']}\n")
            f.write(f"  New non-bomb files created: {aug_info['new_nb']}\n")
            f.write(f"  Final bombs: {aug_info['final_yb']}\n")
            f.write(f"  Final non-bombs: {aug_info['final_nb']}\n")
            f.write(f"  Final ratio (non-bombs:bombs): {aug_info['ratio']:.1f}:1\n\n")
            f.write("NOTES:\n")
            f.write("  - Original files are preserved in augmented directory\n")
            f.write(
                "  - Augmented files use naming: original_aug1.wav, original_aug2.wav, etc.\n"
            )
            f.write("  - All files are 2.88s duration at 8kHz sample rate\n")
            f.write("  - Each augmentation has 50% probability of being applied\n")

        logger.info(f"Metadata saved to: {metadata_file}")


def main():
    """Run the augmentation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply data augmentation to balance training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O directory arguments
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Base data directory"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing training audio files. Default: data-dir/final_new_dataset/train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for augmented files. Default: data-dir/final_new_dataset/train_augmented",
    )

    args = parser.parse_args()

    logger.info("Starting augmentation pipeline...")

    augmenter = AudioAugmenter(
        data_dir=args.data_dir, input_dir=args.input_dir, output_dir=args.output_dir
    )
    aug_info = augmenter.apply_augmentation()

    if aug_info:
        augmenter.create_metadata_file(aug_info)
        logger.info("Augmentation complete!")
    else:
        logger.error("Failed to complete augmentation!")


if __name__ == "__main__":
    main()
