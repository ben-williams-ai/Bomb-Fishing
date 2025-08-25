# Bomb Detection Model Retraining Scripts

This directory contains all the scripts needed to retrain the bomb detection model from scratch, from raw data processing to final model training and evaluation.

## Quick Start

```bash
# 1. Set up environment (UV - Recommended)
cd /path/to/Bomb-Fishing
# UV automatically manages the environment - no activation needed!

# 2. Run complete pipeline
cd retraining_scripts
uv run python data_preprocessing.py
uv run python apply_augmentation.py  
uv run python create_train_test_split.py
uv run python extract_features.py
uv run python train_model.py

# Or run the complete pipeline in one command:
uv run python run_complete_pipeline.py
```

## Prerequisites

### Environment Setup

Choose one of these environment setup methods:

#### Option 1: UV with pyproject.toml (Recommended)
```bash
# From project root - UV manages everything automatically!
# No virtual environment activation needed
uv run python --version  # Test that UV is working

# All dependencies are automatically installed from pyproject.toml
# when you run scripts with 'uv run'
```

#### Option 2: UV with Manual Environment
```bash
# From project root
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

#### Option 3: Automated UV Script
```bash
# From project root
./rebuild_environment_uv.sh
```

#### Option 4: Conda (Traditional)
```bash
# From project root
conda env create -f set_up/linux_env.yml
conda activate conda-bomb-env
```

### Required Directory Structure

The scripts expect this directory structure:

```
Bomb-Fishing/
├── data/                              # Main data directory
│   ├── compressed_new_data/           # Your compressed (.zip) raw data files
│   │   ├── 2023_aug_21.zip
│   │   ├── 2023_nov_23.zip
│   │   └── ...
│   ├── annotated_spreadsheets/        # CSV annotation files
│   │   ├── south_2023_aug21.csv
│   │   ├── south_2023_nov23.csv
│   │   └── ...
│   ├── extracted/                     # Auto-created during processing
│   ├── processed_new_data/            # Auto-created during processing
│   ├── final_new_dataset/             # Auto-created during split
│   └── raw_audio/                     # Auto-created
│       └── final_dataset/
├── models/                            # Auto-created for trained models
├── logs/                              # Auto-created for training logs
└── retraining_scripts/                # This directory
```

**Required input data:**
- Place your compressed data files (`.zip`) in `data/compressed_new_data/`
- Place corresponding CSV annotation files in `data/annotated_spreadsheets/`

## Step-by-Step Pipeline

All scripts now support configurable input/output directories via command-line arguments. See `--help` for each script.

### Step 1: Data Preprocessing

```bash
# UV (Recommended) - automatically manages dependencies
uv run python data_preprocessing.py

# With custom directories
uv run python data_preprocessing.py \
  --data-dir ../data \
  --input-compressed-dir ../data/compressed_new_data \
  --input-annotations-dir ../data/annotated_spreadsheets \
  --output-dir ../data/processed_new_data

# Alternative: If using activated virtual environment
python data_preprocessing.py --help
```

**What it does:**
- Extracts compressed audio files from `data/compressed_new_data/`
- Processes each month's data according to CSV annotations
- Creates 2.88-second audio windows centered on bomb detections
- Resamples all audio to 8kHz (inference requirement)
- Generates YB (bomb) and NB (non-bomb) labeled files
- Saves processed files to `data/processed_new_data/`

**Expected output:**
```
data/processed_new_data/
├── 2023_aug_21/
│   ├── YB000361_M01_20230803_090800_det8.0s_ann8.6s.wav
│   ├── NB000362_M01_20230803_090805_det13.0s.wav
│   └── ...
├── 2023_nov_23/
└── ...
```

**Verification:**
```bash
# Verify specific files against annotations
python data_preprocessing.py --verify-month 2023_aug_21 --verify-files YB000361_M01_20230803_090800_det8.0s_ann8.6s.wav
```

### Step 2: Data Augmentation

```bash
# UV (Recommended)
uv run python apply_augmentation.py

# With custom directories
uv run python apply_augmentation.py \
  --data-dir ../data \
  --input-dir ../data/final_new_dataset/train \
  --output-dir ../data/final_new_dataset/train_augmented

# Alternative: If using activated virtual environment
python apply_augmentation.py --help
```

**What it does:**
- Applies data augmentation to balance the dataset
- **Bombs (YB)**: 10x augmentation (pitch shift, time stretch, noise, etc.)
- **Non-bombs (NB)**: 2x augmentation
- Uses audiomentations library with same pipeline as original training
- Preserves original files and adds augmented versions

**Augmentation techniques:**
- Gaussian noise addition
- Pitch shifting (±2 semitones)
- Time stretching (0.8x to 1.2x)
- Gain variation
- Parametric EQ
- Clipping distortion

### Step 3: Train/Test Split

```bash
# UV (Recommended)
uv run python create_train_test_split.py

# With custom directories
uv run python create_train_test_split.py \
  --data-dir ../data \
  --input-dir ../data/processed_new_data \
  --output-dir ../data/final_new_dataset

# Alternative: If using activated virtual environment
python create_train_test_split.py --help
```

**What it does:**
- Splits data by month to avoid data leakage
- **Test months**: 2023_aug_21, 2023_nov_23, 2024_march_12
- **Training months**: All others
- Combines with original training data for expanded dataset
- Creates final directory structure for training

**Output structure:**
```
data/final_new_dataset/
├── train/
│   ├── YB_files/
│   └── NB_files/
└── test/
    ├── YB_files/
    └── NB_files/
```

### Step 4: Feature Extraction

```bash
# UV (Recommended)
uv run python extract_features.py

# With custom directories
uv run python extract_features.py \
  --data-dir ../data \
  --input-dir ../data/final_new_dataset \
  --output-dir .

# Alternative: If using activated virtual environment
python extract_features.py --help
```

**What it does:**
- Extracts MFCC features from all audio files
- **Features**: 13 MFCC coefficients, mean-aggregated
- **Window**: 2048 FFT, 512 hop length
- **Input shape**: (13,) per audio file
- Saves features and labels as pickle files for training

**Output files:**
- `train_features_labels_2.pickle` - Training data
- `test_features_labels_2.pickle` - Test data

### Step 5: Model Training

```bash
# UV (Recommended)
uv run python train_model.py

# With custom paths
uv run python train_model.py \
  --train-pickle train_features_labels_2.pickle \
  --test-pickle test_features_labels_2.pickle \
  --models-dir ../models \
  --logs-dir ../logs

# Alternative: If using activated virtual environment
python train_model.py --help
```

**What it does:**
- Loads preprocessed features and labels
- Uses AutoKeras for automated architecture search
- **Search space**: Dense networks optimized for audio classification
- **Training**: 100 epochs with early stopping
- **Validation split**: 15% of training data
- **Class weighting**: Automatic balancing for bomb/non-bomb classes
- Saves both AutoKeras model and retrained best model

**Training outputs:**
- `models/autokeras_best_model.keras` - AutoKeras model (new format)
- `models/retrained_best_model.keras` - Final model for inference (new format)
- `models/training_history.png` - Training curves
- `models/training_metadata.txt` - Training details and metrics
- `logs/` - TensorBoard logs

**Model Format Notes:**
- **New models** (from this retraining pipeline): `.keras` files - use with UV and modern Python
- **Legacy models** (existing `code/model/`): Directory format - use with conda environment

## Configuration Options

### Input/Output Directory Configuration

All scripts now support configurable input and output directories to make them more suitable for research publication:

**Command-line arguments available:**

- `--data-dir`: Base data directory (default: `../data`)
- `--input-dir` / `--input-compressed-dir` / `--input-annotations-dir`: Input directories
- `--output-dir` / `--models-dir` / `--logs-dir`: Output directories

**Examples:**

```bash
# Use custom data directory
uv run python data_preprocessing.py --data-dir /path/to/my/data

# Specify exact input/output paths  
uv run python extract_features.py \
  --input-dir /path/to/dataset \
  --output-dir /path/to/features

# Pipeline with custom directories
uv run python run_complete_pipeline.py --data-dir /custom/data/path
```

**Directory structure flexibility:**
- Scripts automatically create output directories if they don't exist
- Relative paths are resolved relative to the script location
- All paths support both relative and absolute paths

### Memory Optimization

For systems with limited memory, edit these parameters in `train_model.py`:

```python
# In ModelTrainer.__init__(), modify these values:
self.BATCH_SIZE = 16           # Reduce from 32
self.MAX_TRIALS = 10           # Reduce from 50  
self.EPOCHS = 50               # Reduce from 100
```

### Training Parameters

Key parameters you can adjust in `train_model.py`:

```python
# AutoKeras search parameters
max_trials=10,              # Reduce for faster search
epochs=100,                 # Reduce for faster training
validation_split=0.15,      # Validation data percentage

# Early stopping
patience=15,                # Epochs to wait for improvement
min_delta=0.001,           # Minimum improvement threshold
```

### Data Processing Options

In `data_preprocessing.py`:

```python
# Audio parameters
target_sample_rate = 8000   # Must match inference pipeline
window_length = 2.88        # Seconds, matches original training

# Extraction parameters  
tolerance_seconds = 2.0     # Tolerance for matching annotations
```

## Testing and Validation

### Verify Data Processing

```bash
# Check bomb counts
uv run python ../count_bombs.py

# Check file counts  
uv run python ../count_yb_files.py

# Verify specific files
uv run python data_preprocessing.py --verify-month 2023_aug_21 --verify-files YB000361_M01_20230803_090800_det8.0s_ann8.6s.wav
```

### Test Trained Model

#### For New .keras Models (Recommended)

```bash
# Quick inference test with new .keras model
uv run python parent_script.py --model-path ../models/retrained_best_model.keras --input-dir test_data/

# Full evaluation on new .keras model
uv run python eval_model.py --model-path models/retrained_best_model.keras --test-pickle test_features_labels_2.pickle

# Threshold tuning for new .keras model
uv run python tune_threshold.py --model-path models/retrained_best_model.keras --train-pickle combined_train_features_labels.pickle
```

#### For Legacy model/ Directory Models

**Important**: Legacy models (saved as `model/` directories) require the conda environment due to older TensorFlow/AutoKeras compatibility:

```bash
# Set up legacy environment
conda env create -f ../set_up/linux_env.yml -n conda-bomb-env
conda activate conda-bomb-env

# Legacy model evaluation
python eval_model.py --model-path ../code/model --test-pickle test_features_labels_2.pickle

# Legacy threshold tuning  
python tune_threshold.py --model-path ../code/model --train-pickle combined_train_features_labels.pickle

# Legacy batch inference (parent/child scripts)
python parent_script.py  # Edit paths in script first
```

## Expected Performance Metrics

After successful training, you should see:

- **Training Accuracy**: >95%
- **Validation Accuracy**: >90%
- **Test Accuracy**: >85%
- **Precision**: >0.8 (bomb detection)
- **Recall**: >0.9 (bomb detection)

## Troubleshooting

### Common Issues

#### 1. Memory Errors During Training
```bash
# Solution: Reduce batch size or max trials in train_model.py
# Edit the script to set smaller values:
# self.BATCH_SIZE = 16  # Reduce from 32
# self.MAX_TRIALS = 10  # Reduce from 50
```

#### 2. Missing Data Files
```bash
# Check data directory structure
ls -la data/compressed_new_data/
ls -la data/annotated_spreadsheets/
```

#### 3. Environment Issues
```bash
# Test environment with UV
uv run python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
uv run python -c "import autokeras as ak; print(f'AutoKeras: {ak.__version__}')"
uv run python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
```

#### 4. CUDA/GPU Issues (if using GPU)
```bash
# Check GPU availability
uv run python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Log Files

Check these locations for detailed logs:
- **Training logs**: `logs/` directory
- **Script output**: Console output from each script
- **Error logs**: Look for Python tracebacks

### Data Validation

If you suspect data issues:

```bash
# Check individual files
uv run python -c "
import librosa
audio, sr = librosa.load('data/processed_new_data/2023_aug_21/YB000361_M01_20230803_090800_det8.0s_ann8.6s.wav')
print(f'Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz')
"
```

## Resuming Interrupted Runs

The pipeline is designed to be resumable:

- **Data preprocessing**: Skips already extracted files
- **Augmentation**: Skips already augmented files  
- **Feature extraction**: Can be re-run safely
- **Training**: Creates new model each time (intentional)

To force re-processing:
```bash
# Remove intermediate directories
rm -rf data/extracted data/processed_new_data data/final_new_dataset
rm -f *_features_labels_2.pickle

# Then re-run pipeline
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Real-time Monitoring

Training progress is displayed in real-time showing:
- Epoch progress
- Loss and accuracy metrics
- AutoKeras trial results
- Validation performance

## Model Deployment

### For New .keras Models (Recommended)

After successful training:

1. **Model location**: `models/retrained_best_model.keras`
2. **Test inference**: `uv run python parent_script.py --model-path ../models/retrained_best_model.keras --input-dir test_data/`
3. **Batch processing**: Use `parent_script.py` for batch processing multiple files
4. **Production deployment**: Copy `.keras` file to production environment

### For Legacy model/ Directory Models

For existing models in directory format (e.g., `code/model/`):

1. **Environment setup**: `conda activate conda-bomb-env` (using `linux_env.yml`)
2. **Test inference**: `python parent_script.py --model-path ../code/model --input-dir test_data/`
3. **Batch processing**: Use `parent_script.py` and `child_script.py` (edit paths first)
4. **Production deployment**: Copy entire `model/` directory

**Model Format Differences:**
- **New (.keras)**: Single file, modern TensorFlow/AutoKeras, UV compatible
- **Legacy (model/)**: Directory with variables/, saved_model.pb, requires older conda env

## Additional Resources

- **Environment setup**: `../rebuild_environment_uv.sh` - Automated UV setup script
- **Inference guide**: `../INFERENCE_INSTRUCTIONS.md`
- **Project structure**: See main `../README.md` for overview

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your environment setup
3. Check data directory structure
4. Review log files for specific errors

For data-related questions, use the verification commands to check data integrity and processing results.

---

## Legacy Model Support

This directory also contains scripts for working with legacy models (saved as `model/` directories).

### Legacy Scripts

#### `parent_script.py` & `child_script.py` - Legacy Batch Inference

**Purpose**: Batch processing system for legacy `model/` directory format models.

**Setup:**
```bash
# Required: conda environment with older TensorFlow/AutoKeras
conda env create -f ../set_up/linux_env.yml -n conda-bomb-env
conda activate conda-bomb-env
```

**Usage:**
```bash
# 1. Edit paths in parent_script.py:
#    - new_audio_dir: Path to input audio files
#    - output_folder: Path for results
#    - model_dir: Path to model/ directory (e.g., "../code/model")
#    - decision_threshold: Detection threshold

# 2. Run batch processing
python parent_script.py
```

**What it does:**
- Processes audio files in configurable batches
- Uses legacy model loading with TensorFlow 2.10.1
- Saves detected bomb clips with timestamps
- Creates CSV results with probabilities and margins
- Handles corrupted files gracefully

#### `eval_model.py` & `tune_threshold.py` - Legacy Model Analysis

These scripts work with both new `.keras` and legacy `model/` formats:

**For legacy models:**
```bash
conda activate conda-bomb-env

# Evaluate legacy model
python eval_model.py --model-path ../code/model --test-pickle test_features_labels_2.pickle

# Tune threshold for legacy model
python tune_threshold.py --model-path ../code/model --train-pickle combined_train_features_labels.pickle
```

**For new .keras models:**
```bash
# Modern approach
uv run python eval_model.py --model-path models/retrained_best_model.keras --test-pickle test_features_labels_2.pickle

uv run python tune_threshold.py --model-path models/retrained_best_model.keras --train-pickle combined_train_features_labels.pickle
```

### Legacy vs New Model Compatibility

| Aspect | Legacy (model/) | New (.keras) |
|--------|----------------|--------------|
| **Format** | Directory with variables/, saved_model.pb | Single .keras file |
| **TensorFlow** | 2.10.1 | 2.19+ |
| **AutoKeras** | 1.1.0 | 2.2.0+ |
| **Environment** | Conda (linux_env.yml) | UV or modern Python |
| **Loading** | `tf.keras.models.load_model()` | `ak.load_model()` or `keras.models.load_model()` |
| **Custom Objects** | Manual CastToFloat32 | Automatic with AutoKeras |

### Migration Notes

- **New models are recommended** for better compatibility and performance
- **Legacy models remain supported** for existing deployments
- **Use the retraining pipeline** to create new .keras models
- **Legacy scripts are provided** for compatibility with existing workflows
