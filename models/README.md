# Models Directory

This directory contains trained models for bomb fishing detection.

## Available Models

### Legacy Model (Production Ready)
- **Location**: `legacy_model/`
- **Format**: TensorFlow SavedModel directory
- **Status**: Production ready - used in original study
- **Usage**: Use with inference scripts in `../inference/`
- **Environment**: Requires conda environment (`conda-bomb-env`)
- **Performance**: Validated on study datasets

### Future Models (Planned)
- **Format**: Modern .keras files
- **Status**: In development
- **Benefits**: Better performance, GPU support, easier deployment
- **Timeline**: Available when retraining pipeline is completed

## Model Details

The legacy model was trained on underwater audio recordings from HydroMoth devices and has been validated for detecting explosive fishing activities. However, **the model may not generalise well to new locations without retraining** on local data.

For best results in new locations, consider retraining the model using the experimental pipeline in `../retraining_scripts/`.

## File Structure

```
models/
├── legacy_model/              # Original study model
│   ├── saved_model.pb         # Model architecture and weights
│   ├── keras_metadata.pb      # Keras metadata
│   └── variables/             # Model variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
└── README.md                  # This file
```
