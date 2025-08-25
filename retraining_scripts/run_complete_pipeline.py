#!/usr/bin/env python3
"""
Complete Pipeline Runner
Executes the full retraining pipeline from start to finish
"""

import subprocess
import sys
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline_run.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_script(script_name, description, use_uv=True):
    """Run a script and handle errors."""
    logger.info(f"Starting: {description}")

    if use_uv:
        cmd = ["uv", "run", "python", script_name]
        logger.info(f"Running: uv run python {script_name}")
    else:
        cmd = [sys.executable, script_name]
        logger.info(f"Running: python {script_name}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        duration = time.time() - start_time
        logger.info(f"✅ Completed: {description} ({duration:.1f}s)")

        # Log key output lines
        if result.stdout:
            output_lines = result.stdout.strip().split("\n")
            important_lines = [
                line
                for line in output_lines
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "complete",
                        "saved",
                        "total",
                        "accuracy",
                        "error",
                        "warning",
                    ]
                )
            ]
            for line in important_lines[-5:]:  # Last 5 important lines
                logger.info(f"  {line}")

        return True

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"❌ Failed: {description} ({duration:.1f}s)")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_prerequisites():
    """Check if required directories and files exist."""
    logger.info("Checking prerequisites...")

    required_dirs = [
        Path("../data/compressed_new_data"),
        Path("../data/annotated_spreadsheets"),
    ]

    for dir_path in required_dirs:
        if not dir_path.exists() or not any(dir_path.iterdir()):
            logger.error(f"Required directory missing or empty: {dir_path}")
            return False

    # Check for data files
    compressed_files = list(Path("../data/compressed_new_data").glob("*.tar.gz"))
    csv_files = list(Path("../data/annotated_spreadsheets").glob("*.csv"))

    if not compressed_files:
        logger.error(
            "No compressed data files (.tar.gz) found in data/compressed_new_data/"
        )
        return False

    if not csv_files:
        logger.error("No CSV annotation files found in data/annotated_spreadsheets/")
        return False

    logger.info(
        f"Found {len(compressed_files)} compressed files and {len(csv_files)} CSV files"
    )
    return True


def check_uv_available():
    """Check if UV is available and suggest fallback."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        logger.info("✅ UV found - using UV for dependency management")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("⚠️  UV not found - falling back to regular Python")
        logger.warning(
            "Install UV for better performance: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
        return False


def main():
    """Run the complete retraining pipeline."""
    logger.info("🚀 Starting Complete Bomb Detection Model Retraining Pipeline")
    logger.info("=" * 70)

    # Check if UV is available
    use_uv = check_uv_available()

    # Check prerequisites
    if not check_prerequisites():
        logger.error(
            "Prerequisites check failed. Please ensure data files are in place."
        )
        sys.exit(1)

    # Pipeline steps
    steps = [
        (
            "data_preprocessing.py",
            "Data Preprocessing - Extract and process raw audio files",
        ),
        (
            "apply_augmentation.py",
            "Data Augmentation - Apply 10x bomb, 2x non-bomb augmentation",
        ),
        (
            "create_train_test_split.py",
            "Train/Test Split - Create final dataset structure",
        ),
        (
            "extract_features.py",
            "Feature Extraction - Generate MFCC features for training",
        ),
        (
            "train_model.py",
            "Model Training - Train AutoKeras model with extracted features",
        ),
    ]

    start_time = time.time()

    for i, (script, description) in enumerate(steps, 1):
        logger.info(f"\n{'='*20} STEP {i}/5 {'='*20}")

        if not run_script(script, description, use_uv=use_uv):
            logger.error(f"Pipeline failed at step {i}: {description}")
            logger.error("Check the logs above for details.")
            sys.exit(1)

    # Pipeline completion
    total_duration = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Total time: {total_duration/60:.1f} minutes")
    logger.info("=" * 70)

    # Check outputs
    model_path = Path("../models/retrained_best_model.keras")
    if model_path.exists():
        logger.info(f"✅ New .keras model available at: {model_path}")
        logger.info(f"   This model works with UV and modern Python environments")
    else:
        logger.warning("⚠️  Trained model not found at expected location")

    # Next steps
    logger.info("\n📋 Next Steps:")
    if use_uv:
        logger.info(
            "1. Test your model: uv run python parent_script.py --model-path ../models/retrained_best_model.keras --input-dir test_data/"
        )
        logger.info("2. Run evaluation: uv run python eval_model.py")
        logger.info("3. Tune threshold: uv run python tune_threshold.py")
    else:
        logger.info(
            "1. Test your model: python parent_script.py --model-path ../models/retrained_best_model.keras --input-dir test_data/"
        )
        logger.info("2. Run evaluation: python eval_model.py")
        logger.info("3. Tune threshold: python tune_threshold.py")
    logger.info("4. Check training logs: tensorboard --logdir ../logs/")


if __name__ == "__main__":
    main()
