# Tabular ML Benchmark (Unified + TabPFN baseline)

## Overview
- Main pipeline: one unified model for CoverType, HELOC, and HIGGS (adds `dataset_id` feature, shared hyperparameters).
- TabPFN v2.5 baseline: trained separately per dataset (TabPFN max 10 classes, so not unified).

## Setup
- Python 3; install deps: `pip install -r requirements.txt`
- Data CSVs live in `data/raw/`

## Key Files
- `src/train.py` — unified training (rf/lgbm/xgb/…); TabPFN is blocked here due to 11 classes.
- `src/predict.py` — unified prediction + combined submission.
- `src/models_tabular.py` — model definitions; TabPFN default version set to v2.5.
- `src/data_utils.py` — loaders + unified feature/label mapping.
- `baseline.py` — TabPFN v2.5 baseline, per-dataset training/prediction.

## Outputs
- Models: `models/unified_<model>_model.txt` (LGBM) or `.pkl` (others); TabPFN per-dataset models at `models/<dataset>_tabpfn_model.pkl`.
- Metadata: `models/unified_<model>_metadata.json`
- Error analysis: `models/unified_<model>_error_analysis.json`
- Submissions: `submissions/<model>_<dataset>_test_submission.csv` plus combined file.

## How to Run (Unified models: rf/lgbm/xgb/…)
- One-command entry point: `python run.py` (unified pipeline: train + predict; supports `--model`, `--no-cv`, `--train-only`, `--predict-only`).
- Train (with CV): `python src/train.py --dataset unified --model lgbm`
- Train (no CV): `python src/train.py --dataset unified --model lgbm --no-cv`
- Predict: `python src/predict.py --dataset unified --model lgbm`
  - Swap `lgbm` for `rf`/`xgb` as needed.
- Entry point: `python run.py` (unified only).

## TabPFN v2.5 Baseline (per dataset, no downsampling)
- Ensure: `pip install tabpfn torch`; env `TABPFN_MODEL_VERSION` defaults to `v2.5`.
- Train + predict all three: `python baseline.py --no-cv`
- Train only: `python baseline.py --train-only --no-cv`
- Predict only: `python baseline.py --predict-only`
  - Saves three submissions and combined file.

## Notes
- LightGBM models saved as text boosters to avoid pickle segfaults.
- TabPFN not allowed in unified mode (11 classes > 10 limit); use `baseline.py` instead.
