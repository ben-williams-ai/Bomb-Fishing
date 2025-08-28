# Inference Scripts

This directory contains production-ready scripts for running bomb fishing detection on new audio data.

## ** Inference Scripts**

### Quick Start

1. **Set up environment**:
   ```bash
   # For .keras models (modern)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cd inference
   uv run python inference_parent.py
   
   # For legacy TensorFlow SavedModel
   conda env create -f ../set_up/linux_env.yml
   conda activate conda-bomb-env
   cd inference
   python inference_parent.py
   ```

2. **Edit paths in `inference_parent.py`**:
   ```python
   new_audio_dir = Path("/path/to/your/audio/files")        # Your WAV files
   output_folder = Path("/path/to/output/directory")        # Results location
   model_dir = Path("../models/legacy_model")               # For legacy model
   # OR
   model_dir = Path("/path/to/your/model.keras")           # For .keras models
   decision_threshold = 0.8996943831443787                  # Tune as needed
   ```

3. **Run inference**:
   ```bash
   python inference_parent.py
   ```

### Enhanced Features
- **Flexible Model Support**: Works with both legacy SavedModel and modern .keras files
- **Rich Output**: CSV includes File, Timestamp, Probability, and Confidence Margin
- **Configurable Threshold**: Adjust decision threshold for your use case
- **Cross-Platform**: Uses pathlib for Windows/Mac/Linux compatibility
- **Better Error Handling**: Detailed error messages and progress tracking

## **Legacy Scripts (Archived)**

The old legacy `parent_script.py` and `child_script.py` can be found in `../archive/model_deployment/`. These scripts:
- Only work with TensorFlow SavedModel format
- Produce basic CSV output (File, Timestamp only)
- Use fixed 0.5 threshold
- Have Windows-specific path handling

**Migration Note**: The modern scripts provide all the same functionality plus additional features. Update your workflows to use `inference_parent.py` instead.

## Files

- `inference_parent.py` - **🎯 MAIN SCRIPT** - Edit paths here, then run
- `inference_child.py` - Child process (called automatically)
- `README.md` - This file

## Requirements

### For Legacy Models
- Audio files: WAV format, 8kHz sample rate
- Model: TensorFlow SavedModel in `../models/legacy_model/`
- Environment: conda environment with TensorFlow 2.10

### For Modern .keras Models
- Audio files: WAV format, 8kHz sample rate  
- Model: .keras file with AutoKeras layers
- Environment: UV with modern TensorFlow/AutoKeras

## Output

Results are saved as CSV files with enhanced information:
- **File**: Audio filename where bomb fishing was detected
- **Timestamp**: Time (HH:MM:SS) of detection event
- **Probability**: Model confidence score (0.0 - 1.0)
- **Margin**: How much above threshold (probability - threshold)

Individual audio clips of suspected events are also saved for manual verification.