# Bomb-Fishing
Original scripts used to train and deploy the legacy model in Williams et al., (2025).


## Set up

Install conda and set up the environment:
```bash
cd archive
conda env create -f bomb-detector-env.yml
conda activate conda-bomb-env
```

## Training
The `model_training` Nntebooks were followed:
1. `preprocessing_audio`: Set sample rate and slice annotations to size.
2. `augmentations`: Apply the augmentations, storing new augmented audio files.
3. `feature_extraction`: Extract mfcc spectrograms from all data, split into train and test sets, store in pickle files.
4. `autokeras`: Run the model searc and final training on data.
5. `evaluating_performance`: Assess performance of model
6. `inference`: A short test notebook for inferencing on larger datasets for sanity checks. Now replaced by `model_deployment`.

## Deployment:
Deployment was performed using the legacy model and the `model_deployment` scripts. Each month raw HydroMoth files were put ina  directory and the model was used to inference on this, outputting detections to a new directory. See the study for more detail. This process was run using `parent_script.py`, where model, data and outputs paths at the top of this script must be modified.

# Misc
Contains miscellaneous scripts used for the manuscript, such as plots and the acoustic localisation analysis.