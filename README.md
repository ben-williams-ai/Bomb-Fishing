# Bomb Fishing Classifier

A machine learning pipeline for detecting explosive fishing activities in underwater audio recordings. This system helps protect marine ecosystems by automatically identifying illegal blast fishing practices. 

The code in this repository was used to support the forthcoming publication: Widespread bomb fishing in Indo-pacific archipelago revealed through machine learning accelerated passive acoustic monitoring, Williams et al., (2025). See the study for further details.

The original code to develop and deploy the model can be found in `archive`. As part of the ongoing development of this work, much of this code is being refactored and the model improved to support others in using this approach.

**Important Note**: Do not expect our model to generalise well to new locations without retraining it. Please consider retraining the model on your own data. Our code in development in the `retraining_scripts` directory may help with this.

## Inference (Production Ready)

Use our pre-trained legacy model to detect bomb fishing events in raw HydroMoth audio recordings (sampled at 8 kHz).

### Quick Start for Inference

1. **Set up environment** (see [Environment Setup](#environment-setup))
2. **Prepare your data**: Place WAV files in a directory
3. **Run inference**: Use the legacy scripts with the pre-trained model

### Running Inference 

**Inference (supported)**: Use the inference scripts that support both the legacy and retrained models:

```bash
# Navigate to inference directory
cd inference

# Edit inference_parent.py to set your paths:
# - new_audio_dir: path to your WAV files  
# - output_folder: where results should be saved
# - model_dir: path to ../models/legacy_model (or your .keras file)
# - decision_threshold: tune detection sensitivity

# For legacy model (requires conda)
conda env create -f set_up/linux_env.yml
conda activate conda-bomb-env
python inference_parent.py

# For modern .keras models (requires UV)
uv run python inference_parent.py
```

**Legacy Scripts**: The older legacy `parent_script.py` and `child_script.py` have been archived. Use `inference_parent.py` for better features and compatibility.

### Output

The inference will generate audio files of suspected bomb fishing incidents for manual verification alongside CSV files with:
- **File names** where events were detected
- **Timestamps** (HH:MM:SS) of detected events  
- **Probability scores** showing model confidence
- **Confidence margins** (how much above threshold)

## Retraining (In Development)

**Status**: Currently in development - use with caution

A complete pipeline to retrain models with your own annotated data. This is actively being improved and may have bugs or incomplete features.

### Development Setup for Retraining

```bash
# Install UV package manager (recommended for development)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run retraining pipeline (experimental)
cd retraining_scripts
uv run python run_complete_pipeline.py
```

**Documentation for Retraining**:
- **Quick Start**: [retraining_scripts/QUICK_START.md](retraining_scripts/QUICK_START.md) 
- **Full Guide**: [retraining_scripts/README.md](retraining_scripts/README.md)

**Future Improvements** (in development):
- GPU support for inference
- Modern .keras model format
- Improved model architecture
- Better preprocessing pipeline

## Environment Setup

### For Inference (Legacy Model) - Required

```bash
# Create conda environment with exact dependencies
conda env create -f set_up/linux_env.yml
conda activate conda-bomb-env
```

### For Development/Retraining - Experimental

#### Option 1: UV (Recommended for Development)
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# UV automatically manages dependencies from pyproject.toml
uv run python --version  # Test that UV works

# GPU support:
uv sync --group gpu
uv run --group gpu python retraining_scripts/train_model.py

# CPU support:
uv sync --group cpu  
uv run --group cpu python retraining_scripts/train_model.py

# Default (auto-detects available hardware)
uv sync
uv run python retraining_scripts/train_model.py
```

#### Option 2: UV with Manual Environment
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```



## Project Structure

```
Bomb-Fishing/
├── inference/                # Production-ready inference scripts
│   ├── inference_parent.py      # Main inference script (EDIT PATHS HERE)
│   ├── inference_child.py       # Child process for batch processing
│   └── README.md                # Inference documentation
├── models/                   # Trained models
│   └── legacy_model/            # Original study model (TensorFlow SavedModel)
│       ├── saved_model.pb
│       ├── keras_metadata.pb
│       └── variables/
├── archive/                  # Original study code & data
│   ├── model_training/          # Jupyter notebooks from original study
│   ├── model_deployment/        # Legacy deployment scripts and code
│   ├── misc/                    # Analysis scripts and notebooks from study
│   └── README.md                # Original study documentation
├── set_up/                   # Environment configurations
│   ├── linux_env.yml           # Conda environment for legacy model inference
│   └── get_gpu_working.md       # GPU setup guide
├── retraining_scripts/       # Development - model retraining pipeline
│   ├── README.md                # Experimental - use with caution
│   ├── QUICK_START.md
│   ├── run_complete_pipeline.py
│   ├── data_preprocessing.py
│   ├── extract_features.py
│   ├── train_model.py
│   ├── eval_model.py
│   └── tune_threshold.py
├── misc/                     # Utility scripts and notebooks
├── data/                     # Your datasets (create during setup)
└── logs/                     # Training logs and outputs
```

## Getting Started Checklist

### For Inference Only
- [ ] **Install Conda**: Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)
- [ ] **Create Environment**: `conda env create -f set_up/linux_env.yml`
- [ ] **Activate Environment**: `conda activate conda-bomb-env`
- [ ] **Edit Paths**: Update `inference/inference_parent.py` with your paths
- [ ] **Prepare Audio**: Place WAV files (8kHz) in your input directory
- [ ] **Run Inference**: `cd inference && python inference_parent.py`

### For Development/Retraining (Experimental)
- [ ] **Install UV**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] **Prepare Data**: See [retraining_scripts/README.md](retraining_scripts/README.md)
- [ ] **Run Pipeline**: `cd retraining_scripts && uv run python run_complete_pipeline.py`
- [ ] **Note**: This is experimental - expect issues and incomplete features

## Model Information

| Model | Status | Format | Environment | Use Case |
|-------|--------|--------|-------------|----------|
| **Legacy Model** | ✅ Production | TensorFlow SavedModel | Conda + TF 2.10 | Inference on new data |
| **New Models** | ⚠️ In Development | .keras (planned) | UV + Python 3.9+ | Future improved models |

## Development Status

- **Production Ready**: Inference with legacy model
- **In Development**: Model retraining pipeline
- **Planned**: GPU support, improved models, modern .keras format

The retraining scripts and modern model formats are actively being developed. Use the legacy inference pipeline for reliable results.

## Citation

If you use this code in your research, please cite:

```
Williams et al., (2025). Widespread bomb fishing in Indo-pacific archipelago revealed through 
machine learning accelerated passive acoustic monitoring. [Journal details pending publication]
```

## License

See [LICENSE](LICENSE) for details.

---

**Need Help?** 
- For inference issues: Check that your audio is 8kHz WAV format and paths are correctly set
- For development questions: See the experimental documentation in `retraining_scripts/`
- For general questions: Please refer to the original study (link pending publication)