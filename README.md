# Unified Tabular Learning — UvA AML2025 Final Project

## 1. Project Overview
Brief description of:
- The goal of Project 3: classify multiple tabular datasets with a unified model.
- The Kaggle competition setup.
- Our team name, team members, and chosen approach.

## 2. Research Question
A clear, concrete research question that can be answered with “yes/no” or a number.
Example:
- *Can a unified transformer-based encoder outperform simple baselines across multiple tabular datasets?*

## 3. Datasets
Describe:
- The datasets provided in the Kaggle competition.
- Number of datasets, number of features, number of classes.
- Preprocessing steps:
  - Missing values
  - Normalization / standardization
  - Categorical encoding
  - Train/val/test splits

## 4. Baseline Model (Pretrained Model from HuggingFace)
Include:
- Description of chosen pretrained model (name, link).
- Why it is suitable for tabular learning.
- How we fine-tuned it.
- Submission naming (“baseline” on Kaggle).

## 5. Our Model
Describe your custom model:
- Model architecture overview
- Diagram (referenced image file)
- How a single sample flows through the architecture
- Key design choices (activation, normalization, embeddings, regularization)
- Differences from the baseline

## 6. Training Setup
- Hardware used
- Training hyperparameters
  - learning rate
  - batch size
  - epochs
  - optimizer
  - scheduler
- Loss function
- Evaluation metric (e.g., accuracy)

## 7. Results
### 7.1 Quantitative Comparison
Table comparing:
- Simple baseline(s) (e.g., MLP, Logistic Regression)
- Complex baseline (HuggingFace pretrained)
- Our model

Include:
- Accuracy
- Loss
- Kaggle leaderboard score

### 7.2 Computational Complexity Analysis
Compare:
- Number of parameters
- FLOPs (if computed)
- Training time
- Inference speed
- GPU/CPU memory usage

Connect back to the question:
> Baseline may perform better, but at what compute cost?

## 8. Error Analysis
Show 1–2 concrete examples where:
- The model predicts the wrong class
- Explain why (hypothesis)
  - class imbalance
  - noisy features
  - overlapping feature distributions
  - model underfitting or overfitting

Include small tables or visualization.

## 9. Ablation Studies (Optional)
If you tried:
- different feature encodings
- removing/adding layers
- trying different optimizers or regularization  
then briefly summarize.

## 10. Poster Summary
A short version of what will go into the final poster:
- Problem statement
- Model architecture figure
- Main results
- Comparison to baselines
- Key insights
- Error analysis examples

## 11. Folder Structure
Explain your repository structure. Example:

project/
│── data/
│ ├── dataset_1.csv
│ ├── dataset_2.csv
│ └── ...
│── models/
│ ├── baseline_model.py
│ ├── our_model.py
│ └── utils.py
│── training/
│ ├── train_baseline.py
│ ├── train_custom.py
│ └── eval.py
│── notebooks/
│ └── exploration.ipynb
│── submissions/
│ ├── baseline.csv
│ └── custom_model.csv
│── README.md


## 12. How to Run
Include commands such as:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train baseline
python training/train_baseline.py

# 3. Train our model
python training/train_custom.py

# 4. Generate Kaggle submission
python training/eval.py

## 13. References
