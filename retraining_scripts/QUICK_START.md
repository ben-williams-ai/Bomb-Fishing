# Quick Start Guide - Model Retraining

## 5-Minute Setup

```bash
# 1. Environment setup with UV (Recommended - Zero Config!)
# UV automatically manages dependencies from pyproject.toml
uv run python --version  # Test that UV works

# 2. Prepare data structure
mkdir -p data/{compressed_new_data,annotated_spreadsheets}
# Place your .tar.gz files in data/compressed_new_data/
# Place your .csv files in data/annotated_spreadsheets/

# 3. Run complete pipeline (UV automatically installs dependencies)
cd retraining_scripts
uv run python run_complete_pipeline.py

# OR run step by step:
uv run python data_preprocessing.py && uv run python apply_augmentation.py && uv run python create_train_test_split.py && uv run python extract_features.py && uv run python train_model.py
```

## Why UV?

- **Zero Configuration**: No virtual environment setup needed
- **Automatic Dependencies**: Installs packages from pyproject.toml automatically  
- **Fast**: Much faster dependency resolution than pip/conda
- **Reliable**: Consistent environments across machines

## Required Files

**Before starting, ensure you have:**

```
data/compressed_new_data/
├── 2023_aug_21.tar.gz
├── 2023_nov_23.tar.gz
└── [other month files]

data/annotated_spreadsheets/
├── south_2023_aug21.csv
├── south_2023_nov23.csv
└── [other annotation files]
```

## Expected Timeline

| Step | Script | Time | Output |
|------|--------|------|--------|
| 1 | `data_preprocessing.py` | 10-30 min | Processed audio files |
| 2 | `apply_augmentation.py` | 20-60 min | Augmented dataset |
| 3 | `create_train_test_split.py` | 2-5 min | Train/test split |
| 4 | `extract_features.py` | 5-15 min | Feature pickle files |
| 5 | `train_model.py` | 1-3 hours | Trained model |

**Total time: 2-4 hours** (depending on data size and hardware)

## Success Indicators

After each step, you should see:

1. **Data preprocessing**: Files in `data/processed_new_data/[month]/`
2. **Augmentation**: 10x more YB files, 2x more NB files
3. **Train/test split**: Files in `data/final_new_dataset/train/` and `test/`
4. **Feature extraction**: `train_features_labels_2.pickle` and `test_features_labels_2.pickle`
5. **Training**: `models/retrained_best_model.keras` with >85% accuracy

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Memory error during training | Reduce batch size in `train_model.py`: set `BATCH_SIZE = 16` and `MAX_TRIALS = 10` |
| Missing data files | Check file paths and ensure data is in correct directories |
| Environment issues | Check UV is installed: `uv --version`, test: `uv run python --version` |
| Low accuracy (<80%) | Check data quality, verify augmentation worked, try longer training |

## Quick Test

### For New .keras Models (from this pipeline)
```bash
# Test your newly trained model with UV - use parent_script.py for inference
uv run python parent_script.py --model-path ../models/retrained_best_model.keras --input-dir test_data/

# Expected output: Detection results with confidence scores
```

### For Legacy model/ Directory Models
```bash
# Test legacy models (requires conda environment)
conda activate conda-bomb-env  # using ../set_up/linux_env.yml
python parent_script.py --model-path ../code/model --input-dir test_data/

# Or use legacy batch processing with edited paths
python parent_script.py  # Edit paths in script first
```

## Checklist

- [ ] Environment set up and tested
- [ ] Data files in correct directories
- [ ] Data preprocessing completed
- [ ] Augmentation applied
- [ ] Train/test split created
- [ ] Features extracted
- [ ] Model trained successfully (.keras file created)
- [ ] Model tested on sample audio

## Model Format Info

This pipeline creates **new .keras models** that work with UV and modern Python. For **legacy model/ directory models**, use the conda environment and legacy scripts.

For detailed instructions, see [README.md](README.md).
