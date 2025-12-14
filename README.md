# Tabular ML Benchmark (Unified Model)

## Overview
Single, unified model trained across three tabular datasets: CoverType, HELOC, and HIGGS. All training/inference uses the unified feature space (adds `dataset_id`) and one set of hyperparameters per model type.

## Setup
- Python 3. Install deps: `pip install -r requirements.txt`
- Data CSVs are under `data/raw/`

## Key Files
- `src/train.py` — unified training
- `src/predict.py` — unified prediction + combined submission
- `src/models_tabular.py` — model definitions (RF, LGBM, XGB, etc.)
- `src/data_utils.py` — data loaders + unified feature/label mapping

## Outputs
- Models: `models/unified_<model>_model.txt` (LGBM) or `.pkl` (others)
- Metadata: `models/unified_<model>_metadata.json`
- Error analysis: `models/unified_<model>_error_analysis.json`
- Submissions: `submissions/<model>_<dataset>_test_submission.csv` and combined file

## How to Run (unified only)
- Train (with CV): `python src/train.py --dataset unified --model lgbm`
- Train (no CV): `python src/train.py --dataset unified --model lgbm --no-cv`
- Predict: `python src/predict.py --dataset unified --model lgbm`
  - Replace `lgbm` with `rf`/`xgb` if desired

## Notes
- LightGBM is saved as a text booster to avoid pickle issues.
- Hyperparameter search and per-dataset training are removed to comply with the unified-model requirement.
