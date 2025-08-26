# Bomb Fishing Classifier


A machine learning pipeline for detecting explosive fishing activities in underwater audio recordings. This system helps protect marine ecosystems by automatically identifying illegal blast fishing practices.

## Quick Start
```bash
# Complete retraining pipeline with your own data
cd retraining_scripts
uv run python run_complete_pipeline.py

# With custom data directory
uv run python run_complete_pipeline.py --data-dir /path/to/your/data
```
**Full Guide**: [retraining_scripts/README.md](retraining_scripts/README.md)

### For Inference Only (Using Pre-trained Models)

#### Modern .keras Models (Recommended)
```bash
# Run inference with trained model
cd retraining_scripts
uv run python parent_script.py --model-path ../models/your_model.keras --input-dir /path/to/audio/files

# With specific output directory
uv run python parent_script.py --model-path ../models/your_model.keras --input-dir /path/to/audio --output-dir results/
```

#### Legacy Models (Directory Format)
```bash
# Requires conda environment setup
conda activate conda-bomb-env
cd retraining_scripts
python parent_script.py --model-path ../code/model --input-dir /path/to/audio/files
```

## Project Structure

```
Bomb-Fishing/
├── retraining_scripts/       # Retraining pipeline
│   ├── README.md             # Complete retraining documentation
│   ├── QUICK_START.md        # 5-minute setup guide
│   ├── run_complete_pipeline.py  # One-click full pipeline
│   ├── data_preprocessing.py     # Audio data processing
│   ├── apply_augmentation.py     # Data augmentation
│   ├── create_train_test_split.py # Dataset splitting
│   ├── extract_features.py       # MFCC feature extraction
│   ├── train_model.py            # AutoKeras model training
│   ├── eval_model.py             # Model evaluation
│   └── tune_threshold.py         # Decision threshold optimization
├── code/                     # Legacy inference infrastructure
│   └── model/                # Legacy model (directory format)
├── models/                   # Modern trained models (.keras files)
├── set_up/                   # Environment configurations
│   └── linux_env.yml           # Conda environment for legacy models
└── data/                     # Your datasets (created during setup)
    ├── compressed_new_data/     # Raw compressed audio files
    ├── annotated_spreadsheets/  # CSV annotation files
    └── processed_new_data/      # Processed audio (auto-generated)
```

## Environment Setup

### Option 1: UV (Recommended - Zero Config)
```bash
# Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# UV automatically manages all dependencies from pyproject.toml
# No virtual environment setup needed
uv run python --version  # Test that UV works

# Dependencies are automatically installed when you run scripts
uv run python retraining_scripts/parent_script.py --help
```
### Option 2: UV with Manual Environment
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### Option 3: Automated UV Setup
```bash
./rebuild_environment_uv.sh
```

### Option 4: Conda (Traditional)
```bash
conda env create -f set_up/linux_env.yml
conda activate conda-bomb-env
```



## Model Formats

| Model Type | Format | Environment | Usage |
|---------------|-----------|----------------|----------|
| **Modern (.keras)** | Single `.keras` file | UV + Python 3.9+ | `uv run python retraining_scripts/parent_script.py --model-path model.keras` |
| **Legacy (directory)** | `model/` folder | Conda + TF 2.10 | `conda activate env && python retraining_scripts/parent_script.py --model-path model/` |

## Getting Started Checklist

### Data Setup
- [ ] **Create Data Directories**: 
  ```bash
  mkdir -p data/compressed_new_data data/annotated_spreadsheets
  ```
- [ ] **Prepare Data**: Place `.zip` files in `data/compressed_new_data/`
- [ ] **Add Annotations**: Place `.csv` files in `data/annotated_spreadsheets/`
- [ ] **Ensure Matching Names**: CSV and ZIP files must have identical names
  - Example: `north_2024_feb_22.zip` ↔ `north_2024_feb_22.csv`
  - Format: `[region]_[YYYY]_[MMM]_[DD]` or `[region]_[YYYY]_[MMM][DD]`

### Environment Setup  
- [ ] **Install UV**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] **macOS Users**: Update `pyproject.toml` dependencies:
  ```toml
  "tensorflow-macos>=2.16.0",
  "tensorflow-metal>=0.8.0",
  ```

### Pipeline Execution
- [ ] **Run Pipeline**: `cd retraining_scripts && uv run python run_complete_pipeline.py`
- [ ] **Evaluate Results**: Check `models/` for trained `.keras` files



