# Bomb-Fishing
Original scripts used to train and deploy the legacy model in Williams et al., (2025). 

## Set up
The deployment scripts can be run locally. Install conda and set up the environment:
```bash
conda env create -f ../set_up/linux_env.yml
conda activate conda-bomb-env
```
Training was mostly performed on Google colab in early 2023. Given colabs environment changes overtime, these may no longer work natively on colab.

## Training
The `model_training` Notebooks were followed:
1. `1_preprocessing_audio`: Set sample rate and slice annotations to size.
2. `2_augmentations`: Apply the augmentations, storing new augmented audio files.
3. `3_feature_extraction`: Extract mfcc spectrograms from all data, split into train and test sets, store in pickle files.
4. `4_autokeras`: Run the model search and final training on data.
5. `5_evaluating_performance`: Assess performance of model
6. `6_inference`: A short test notebook for inferencing on larger datasets for sanity checks. Now replaced by `model_deployment`.

## Deployment:
Deployment was performed using the legacy model and the `model_deployment` scripts. Each month raw HydroMoth files were put into a directory and the model was used to inference on this, outputting detections to a new directory. See the study and supplementary for more details on how this works. 

### Legacy Deployment Scripts (model_deployment/)
- `parent_script.py` - Main deployment script (edit paths at top)
- `child_script.py` - Child process for batch processing
- `run_inference.py` - CLI tool for inference 
- `batch_runner.py` - Batch processing utilities

**Note**: These scripts have been superseded by the modern inference scripts in `/inference/`. Use the modern scripts for new deployments as they provide enhanced functionality and better compatibility.

# Misc
Contains miscellaneous scripts used for the manuscript, such as plots and the acoustic localisation analysis.