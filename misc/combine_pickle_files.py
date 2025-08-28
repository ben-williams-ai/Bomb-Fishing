#!/usr/bin/env python3
"""
Combine old and new pickle files for training and testing data
"""

import pickle
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pickle_file(file_path):
    """Load a pickle file and return its contents."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def analyze_pickle_structure(data, name):
    """Analyze the structure of pickle data."""
    logger.info(f"\n=== ANALYZING {name} ===")

    if isinstance(data, tuple):
        logger.info(f"Tuple with {len(data)} elements:")
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                logger.info(
                    f"  Element {i}: numpy array, shape={item.shape}, dtype={item.dtype}"
                )
                if (
                    len(item.shape) == 1 and len(item) < 20
                ):  # Small 1D array, might be labels
                    unique_vals = np.unique(item)
                    logger.info(f"    Unique values: {unique_vals}")
            else:
                logger.info(f"  Element {i}: {type(item)}, value={item}")
    else:
        logger.info(f"Type: {type(data)}")
        if isinstance(data, np.ndarray):
            logger.info(f"Shape: {data.shape}, dtype: {data.dtype}")


def combine_datasets(old_data, new_data, dataset_name):
    """Combine old and new datasets."""
    logger.info(f"\n=== COMBINING {dataset_name} DATA ===")

    # Extract features, labels, and input_shape from both datasets
    if len(old_data) == 3:
        old_features, old_labels, old_input_shape = old_data
    else:
        logger.error(f"Unexpected old data structure: {len(old_data)} elements")
        return None

    if len(new_data) == 3:
        new_features, new_labels, new_input_shape = new_data
    else:
        logger.error(f"Unexpected new data structure: {len(new_data)} elements")
        return None

    logger.info(f"Old {dataset_name}:")
    logger.info(f"  Features: {old_features.shape}")
    logger.info(f"  Labels: {old_labels.shape}")
    logger.info(f"  Input shape: {old_input_shape}")

    logger.info(f"New {dataset_name}:")
    logger.info(f"  Features: {new_features.shape}")
    logger.info(f"  Labels: {new_labels.shape}")
    logger.info(f"  Input shape: {new_input_shape}")

    # Check if input shapes are compatible
    if old_input_shape != new_input_shape:
        logger.warning(f"Input shapes differ: {old_input_shape} vs {new_input_shape}")
        # Use the new input shape as it's likely more recent
        combined_input_shape = new_input_shape
    else:
        combined_input_shape = old_input_shape

    # Combine features and labels
    try:
        combined_features = np.concatenate([old_features, new_features], axis=0)
        combined_labels = np.concatenate([old_labels, new_labels], axis=0)

        logger.info(f"Combined {dataset_name}:")
        logger.info(f"  Features: {combined_features.shape}")
        logger.info(f"  Labels: {combined_labels.shape}")
        logger.info(f"  Input shape: {combined_input_shape}")

        # Count bombs vs non-bombs
        bombs = np.sum(combined_labels)
        non_bombs = len(combined_labels) - bombs
        logger.info(f"  Bombs: {bombs}, Non-bombs: {non_bombs}")

        return combined_features, combined_labels, combined_input_shape

    except Exception as e:
        logger.error(f"Error combining {dataset_name} data: {e}")
        return None


def save_combined_pickle(data, output_path):
    """Save combined data to pickle file."""
    try:
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved combined data to: {output_path}")

        # Get file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"File size: {file_size:.1f} MB")

    except Exception as e:
        logger.error(f"Error saving to {output_path}: {e}")


def main():
    """Main function to combine pickle files."""
    logger.info("Starting pickle file combination...")

    # Define file paths
    old_train_path = "/Users/sonnyburniston/Bomb-Fishing/train_features_labels_2.pickle"
    old_test_path = "/Users/sonnyburniston/Bomb-Fishing/test_features_labels_2.pickle"
    new_train_path = (
        "/Users/sonnyburniston/Bomb-Fishing/data/new_data_train_features_labels.pickle"
    )
    new_test_path = (
        "/Users/sonnyburniston/Bomb-Fishing/data/new_data_test_features_labels.pickle"
    )

    # Output paths
    combined_train_path = (
        "/Users/sonnyburniston/Bomb-Fishing/combined_train_features_labels.pickle"
    )
    combined_test_path = (
        "/Users/sonnyburniston/Bomb-Fishing/combined_test_features_labels.pickle"
    )

    # Load all pickle files
    logger.info("Loading pickle files...")
    old_train_data = load_pickle_file(old_train_path)
    old_test_data = load_pickle_file(old_test_path)
    new_train_data = load_pickle_file(new_train_path)
    new_test_data = load_pickle_file(new_test_path)

    if None in [old_train_data, old_test_data, new_train_data, new_test_data]:
        logger.error("Failed to load some pickle files. Exiting.")
        return

    # Analyze structures
    analyze_pickle_structure(old_train_data, "OLD TRAINING")
    analyze_pickle_structure(old_test_data, "OLD TEST")
    analyze_pickle_structure(new_train_data, "NEW TRAINING")
    analyze_pickle_structure(new_test_data, "NEW TEST")

    # Combine training data
    combined_train = combine_datasets(old_train_data, new_train_data, "TRAINING")
    if combined_train is None:
        logger.error("Failed to combine training data")
        return

    # Combine test data
    combined_test = combine_datasets(old_test_data, new_test_data, "TEST")
    if combined_test is None:
        logger.error("Failed to combine test data")
        return

    # Save combined data
    logger.info("\nSaving combined datasets...")
    save_combined_pickle(combined_train, combined_train_path)
    save_combined_pickle(combined_test, combined_test_path)

    logger.info("\n=== COMBINATION COMPLETE ===")
    logger.info(f"Output files:")
    logger.info(f"  Training: {combined_train_path}")
    logger.info(f"  Test: {combined_test_path}")


if __name__ == "__main__":
    main()
