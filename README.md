# breast-cancer-identification
Identifying high-risk breast cancer using digital pathology images

## Usage

### Split train/test/holdout set

Run `split_biopsies.ipynb`

It will export several CSV files in `csv_dir` folder that record the mapping relationship of downsampled slices information and labels.

### Export Python Pickle format train/test/holdout set

Run `prepare_datasets.ipynb`

The script exports a dictionary for each train/test/holdout set in `datasets/` folder. Like

```python
pd.to_pickle({'x': holdout_x_list, 'y': holdout_y_list, 'id': holdout_biopsy_id_list}, f'./datasets/holdout.pkl')
```

- x: slice tensor. Croped and Normalized to 224x224x3 resolution.
- y: label. {0, 1, 2, 3, 4}
- id: BiopsyID, the slice image belongs to.

### Train Deep learning models

- Run `train_pipeline_resnet50.ipynb` for RestNet-50 model
- Run `train_pipeline_swin.py` for Swin-Large model

Above two scripts will save model parameters in `checkpoints/` folder.

### Ensemble two models' outputs for holdout set

Run `test_ensemble.ipynb`

The script load ResNet and Swin models' parameters and outputs the prediction logits for holdout set.

Then we calibrate and ensemble the two outputs (take average scores) and export the final prediction results for expected holdout set.
