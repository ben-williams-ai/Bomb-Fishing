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
📖 **Full Guide**: [retraining_scripts/README.md](retraining_scripts/README.md)

### 🎯 For Inference Only (Using Pre-trained Models)

#### Modern .keras Models (Recommended)
```bash
# Quick single-file inference
uv run python simple_inference.py models/retrained_best_model.keras test_file.wav

# Batch processing multiple files  
uv run python modern_inference.py
```

#### Legacy Models (Directory Format)
```bash
# Requires conda environment setup
conda activate conda-bomb-env
python simple_inference.py code/model test_file.wav
```

**Inference Guide**: [INFERENCE_INSTRUCTIONS.md](INFERENCE_INSTRUCTIONS.md)

##  Project Structure

```
Bomb-Fishing/
├── retraining_scripts/       # Research-ready retraining pipeline
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
├── simple_inference.py       # Single file inference
├── modern_inference.py       # Batch processing
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
uv run python simple_inference.py --help
```
### Option 2: UV with Manual Environment
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements_new_stack.txt
```

### Option 3: Automated UV Setup
```bash
./rebuild_environment_uv.sh
```

### Option 4: Conda (Traditional)
```bash
conda env create -f set_up/macos_arm64_env_fixed_new.yml
conda activate bomb-audio-env-arm64-new
```



## 🔄 Model Formats

| Model Type | Format | Environment | Usage |
|---------------|-----------|----------------|----------|
| **Modern (.keras)** | Single `.keras` file | UV + Python 3.9+ | `uv run python script.py model.keras` |
| **Legacy (directory)** | `model/` folder | Conda + TF 2.10 | `conda activate env && python script.py model/` |

    ##  Getting Started Checklist

### For New Researchers
- [ ] **Install UV**: `curl -LsSf https://astral.sh/uv/install.sh | sh`  
- [ ] **Prepare Data**: Place `.tar.gz` files in `data/compressed_new_data/`
- [ ] **Add Annotations**: Place `.csv` files in `data/annotated_spreadsheets/`
- [ ] **Run Pipeline**: `cd retraining_scripts && uv run python run_complete_pipeline.py`
- [ ] **Evaluate Results**: Check `models/` for trained `.keras` files



