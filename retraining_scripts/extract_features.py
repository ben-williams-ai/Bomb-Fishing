#!/usr/bin/env python3
"""
Feature extraction for bomb detection model
Extracts MFCC features from audio data
Supports configurable input/output directories
"""

import datetime
import numpy as np
import random
import os
import pickle
from pathlib import Path
from typing import Optional

# Progress tracker for feature extraction
from tqdm import tqdm

# For audio processing
import librosa
import librosa.display

# For visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set up matplotlib
mpl.rcParams["figure.figsize"] = (12, 10)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class FeatureExtractor:

    def __init__(
        self,
        data_dir: str = "../data",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize FeatureExtractor with configurable I/O directories.

        Args:
            data_dir: Base data directory (default: "../data")
            input_dir: Directory containing train/test split data (default: data_dir/final_new_dataset)
            output_dir: Output directory for feature files (default: data_dir)
        """
        self.data_dir = Path(data_dir)
        self.final_dataset_dir = (
            Path(input_dir) if input_dir else self.data_dir / "final_new_dataset"
        )
        self.output_dir = Path(output_dir) if output_dir else self.data_dir

        # Directory paths
        self.train_dir = self.final_dataset_dir / "train"
        self.test_dir = self.final_dataset_dir / "test"

        # Sample rate (matching the processed audio files)
        self.sample_rate = 8000

        # Set random seed for reproducibility
        self.seed = 123
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Output pickle files
        self.train_pickle_file = self.output_dir / "train_features_labels.pickle"
        self.test_pickle_file = self.output_dir / "test_features_labels.pickle"

        # Verify test directory exists
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

    def view_mel_spec(self, filename: str, audio_dir: Path):
        """Creates a spectrogram plot of an audio file in the mel scale.

        Args:
            filename: Audio filename
            audio_dir: Directory containing the audio file
        """
        # Set hop length for mel_spec and specshow()
        hop_length = 64

        # Read in the audio file
        file_path = audio_dir / filename
        audio, sample_rate = librosa.load(file_path, sr=self.sample_rate)

        # Compute the mel spectrogram of the audio
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, win_length=1024, hop_length=hop_length
        )

        # Convert the power spectrogram to decibel (dB) units
        mel_spec_db = librosa.power_to_db(S=mel_spec, ref=1.0)

        # Plot the spectrogram
        plt.figure(figsize=(5, 5))
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sample_rate,
            y_axis="mel",
            x_axis="time",
            vmin=-100,
            vmax=0,
            cmap="magma",
            hop_length=hop_length,
        )
        plt.colorbar(img)

        # Set y-axis
        plt.yticks([200, 500, 1000, 2000, 4000])
        plt.ylim(0, 4000)
        plt.title(f"Mel Spectrogram: {filename}")

    def extract_features_labels(self, dataset: list, audio_dir: Path):
        """Extract MFCC features and labels from audio files.

        Args:
            dataset: List of audio filenames
            audio_dir: Directory containing the audio files

        Returns:
            features: Numpy array of MFCC features
            labels: Numpy array of labels (0 for non-bomb, 1 for bomb)
            input_shape: Shape of input features for the network
        """
        # Create empty lists
        feature_list = []
        label_list = []

        # Extract MFCC from all files
        for file in tqdm(dataset, desc="Extracting features"):
            # Load audio
            audio_path = audio_dir / file
            audio, _ = librosa.load(path=audio_path, sr=self.sample_rate)

            # Calculate MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=32)

            # Define new shape
            shape = np.shape(mfcc)
            input_shape = tuple(list(shape) + [1])

            # Set new shape and append to list
            feature_list.append(np.reshape(mfcc, input_shape))

            # Get label from first character of file name and append to list
            if file[0:2] == "NB":
                label_list.append([0])  # Non-bomb
            elif file[0:2] == "YB":
                label_list.append([1])  # Bomb
            else:
                raise ValueError(f"Unexpected file prefix in {file}")

        # Convert to numpy arrays which can be input into the network
        features = np.array(feature_list)
        labels = np.array(label_list)

        return features, labels, input_shape

    def custom_sort_key(self, string: str):
        """Custom sorting key for test files - NB files first, then YB files."""
        if string.startswith("NB"):
            return (0, int(string[2:7]))
        elif string.startswith("YB"):
            return (1, int(string[2:7]))
        else:
            raise ValueError(f"Unexpected string format: {string}")

    def plot_mfcc(self, features: np.ndarray, chosen_file: int, title: str = "MFCC"):
        """Plot MFCC features for a specific file."""
        # Drop the extra channel added for the network
        spec_data = features[chosen_file][:, :, 0]

        # Create a figure with specified width and height
        fig = plt.figure(figsize=(8, 5))

        # Add subplot to the figure
        ax = fig.add_subplot(111)

        # Plot the data
        ax.imshow(
            spec_data,
            interpolation="nearest",
            cmap="magma",
            origin="lower",
            vmin=-100,
            vmax=0.0,
        )
        ax.set_title(f"{title}: File {chosen_file}")

        # Set aspect ratio to the aspect ratio of the data
        aspect = spec_data.shape[1] / spec_data.shape[0]
        ax.set_aspect(aspect)

        plt.show()

    def extract_test_features(self):
        """Extract features from test data."""
        print("=== EXTRACTING TEST FEATURES ===")

        # List all audio files in the directory
        test_files = [
            f
            for f in os.listdir(self.test_dir)
            if f.endswith(".wav") or f.endswith(".WAV")
        ]
        test_file_count = len(test_files)
        print(f"Found {test_file_count} test files")

        # Sort test files (NB files first, then YB files)
        sorted_test_files = sorted(test_files, key=self.custom_sort_key)

        # Extract features and labels
        test_features, test_labels, input_shape = self.extract_features_labels(
            sorted_test_files, self.test_dir
        )

        # Print statistics
        print(f"\nInput shape for network: {input_shape}")
        bomb_indices = np.where(test_labels == 1)[0]
        non_bomb_indices = np.where(test_labels == 0)[0]
        print(f"Bomb files: {len(bomb_indices)}")
        print(f"Non-bomb files: {len(non_bomb_indices)}")
        print(f"Bomb file indices: {bomb_indices}")

        # Save to pickle file
        pickle_file_path = Path(self.test_pickle_file)
        with open(pickle_file_path, "wb") as f:
            pickle.dump((test_features, test_labels, input_shape), f)

        print(f"Test features saved to: {pickle_file_path}")

        return test_features, test_labels, input_shape

    def load_pickle_file(self, pickle_file: str):
        """Load features and labels from pickle file."""
        pickle_file_path = Path(pickle_file)

        if not pickle_file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_file_path}")

        with open(pickle_file_path, "rb") as f:
            features, labels, input_shape = pickle.load(f)

        return features, labels, input_shape

    def create_metadata(self, test_features, test_labels):
        """Create metadata file documenting the feature extraction process."""
        metadata_file = self.output_dir / "feature_extraction_metadata.txt"

        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write("FEATURE EXTRACTION METADATA\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("FEATURE EXTRACTION PARAMETERS:\n")
            f.write(f"  Sample rate: {self.sample_rate} Hz\n")
            f.write("  MFCC features: 32\n")
            f.write(f"  Random seed: {self.seed}\n\n")

            f.write("DATASET INFORMATION:\n")
            f.write(f"  Total files: {len(test_features)}\n")
            f.write(f"  Bomb files: {np.sum(test_labels)}\n")
            f.write(f"  Non-bomb files: {len(test_labels) - np.sum(test_labels)}\n")
            f.write(f"  Feature shape: {test_features.shape}\n\n")

            f.write("OUTPUT FILES:\n")
            f.write(f"  Test features: {self.test_pickle_file}\n\n")

            f.write("NOTES:\n")
            f.write(
                "  - Features are MFCC spectrograms with shape (32, time_steps, 1)\n"
            )
            f.write("  - Labels: 0 = non-bomb (NB), 1 = bomb (YB)\n")
            f.write("  - Configurable input/output directories via command line\n")
            f.write(f"  - Source: {self.final_dataset_dir}\n")

        print(f"Metadata saved to: {metadata_file}")

    def extract_train_test_features(self):
        """Extract features from both training and test data."""
        print("=== EXTRACTING TRAINING AND TEST FEATURES ===")

        # Check if both train and test directories exist
        has_train_data = self.train_dir.exists() and any(self.train_dir.glob("*.wav"))
        has_test_data = self.test_dir.exists() and any(self.test_dir.glob("*.wav"))

        train_features, train_labels, input_shape = None, None, None
        test_features, test_labels = None, None

        if has_train_data:
            print(f"Extracting training features from: {self.train_dir}")
            train_files = [
                f
                for f in os.listdir(self.train_dir)
                if f.endswith(".wav") or f.endswith(".WAV")
            ]
            train_file_count = len(train_files)
            print(f"Found {train_file_count} training files")

            # Sort files (NB files first, then YB files)
            sorted_train_files = sorted(train_files, key=self.custom_sort_key)

            # Extract features and labels
            train_features, train_labels, input_shape = self.extract_features_labels(
                sorted_train_files, self.train_dir
            )

            print(f"Training features: {train_features.shape}")
            bomb_indices = np.where(train_labels == 1)[0]
            non_bomb_indices = np.where(train_labels == 0)[0]
            print(
                f"Training - Bomb files: {len(bomb_indices)}, Non-bomb files: {len(non_bomb_indices)}"
            )

            # Save training features
            with open(self.train_pickle_file, "wb") as f:
                pickle.dump((train_features, train_labels, input_shape), f)
            print(f"Training features saved to: {self.train_pickle_file}")

        if has_test_data:
            print(f"Extracting test features from: {self.test_dir}")
            test_files = [
                f
                for f in os.listdir(self.test_dir)
                if f.endswith(".wav") or f.endswith(".WAV")
            ]
            test_file_count = len(test_files)
            print(f"Found {test_file_count} test files")

            # Sort files (NB files first, then YB files)
            sorted_test_files = sorted(test_files, key=self.custom_sort_key)

            # Extract features and labels
            test_features, test_labels, test_input_shape = self.extract_features_labels(
                sorted_test_files, self.test_dir
            )

            # Use input shape from training data if available, otherwise use test data shape
            if input_shape is None:
                input_shape = test_input_shape

            print(f"Test features: {test_features.shape}")
            bomb_indices = np.where(test_labels == 1)[0]
            non_bomb_indices = np.where(test_labels == 0)[0]
            print(
                f"Test - Bomb files: {len(bomb_indices)}, Non-bomb files: {len(non_bomb_indices)}"
            )

            # Save test features
            with open(self.test_pickle_file, "wb") as f:
                pickle.dump((test_features, test_labels, input_shape), f)
            print(f"Test features saved to: {self.test_pickle_file}")

        if not has_train_data and not has_test_data:
            raise FileNotFoundError(
                f"No audio files found in {self.train_dir} or {self.test_dir}"
            )

        return train_features, train_labels, test_features, test_labels, input_shape

    def run_feature_extraction(self):
        """Run the feature extraction pipeline."""
        print("Starting feature extraction pipeline...")

        # Extract features from both train and test data
        train_features, train_labels, test_features, test_labels, input_shape = (
            self.extract_train_test_features()
        )

        # Create metadata for test features
        if test_features is not None:
            self.create_metadata(test_features, test_labels)

        print("\n=== FEATURE EXTRACTION COMPLETE ===")
        if train_features is not None:
            print(f"Training features: {train_features.shape}")
        if test_features is not None:
            print(f"Test features: {test_features.shape}")
        print(f"Input shape for network: {input_shape}")

        return train_features, train_labels, test_features, test_labels, input_shape


def main():
    """Run the feature extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract MFCC features from audio data for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O directory arguments
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Base data directory"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing train/test split data. Default: data-dir/final_new_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for feature pickle files. Default: data-dir",
    )

    args = parser.parse_args()

    extractor = FeatureExtractor(
        data_dir=args.data_dir, input_dir=args.input_dir, output_dir=args.output_dir
    )
    extractor.run_feature_extraction()


if __name__ == "__main__":
    main()
