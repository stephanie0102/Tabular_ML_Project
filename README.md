# Tabular ML Benchmark

## Project Overview
This project trains and tests tabular models on three datasets: CoverType, HELOC, and HIGGS. The default baseline uses the TabPFN model. All code is Python.

## Research Question
Can a unified machine learning pipeline achieve better performance over a pretrained baseline on heterogeneous tabular datasets with varying class imbalance?

## Setup
- Python
- Install packages: `pip install -r requirements.txt`.
- Data CSV files are already in `data/raw/`.

## key files
- `run.py`: main entry for training and prediction.
- `baseline.py`: runn for the TabPFN baseline.
- `src/train.py`: training helpers.
- `src/predict.py`: prediction and submission helpers.
- `src/models_tabular.py`: model definitions.
- `src/data_utils.py`: data loading for the three datasets.
 
## Outputs
- Trained models are saved in `models/` as `dataset_model.pkl`.
- Submission CSVs go to `submissions/`.  
- Per-dataset files look like `baseline_covtype_test_submission.csv`.  
- Combined file is `combined_submission.csv`.

## How to run
- Full pipeline with chosen model (`rf`, `lgbm`, `xgb`, `lr`, `mlp`, `ensemble`, `baseline`):  
  `python run.py --model lgbm`
- Train only: `python run.py --model lgbm --train-only`.
- Predict only: `python run.py --model lgbm --predict-only`.
- Turn off cross-validation: add `--no-cv`.
