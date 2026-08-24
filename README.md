# Content Decline Prediction — Applied Search Intelligence

Predicting whether published content is currently declining in organic search performance, using 90-day engagement and keyword signals from the FlyRank content-refresh dataset.

**Author:** Ishaq Ahmad Khan · Machine Learning Engineer · 2026

## Problem

A content team managing hundreds or thousands of published pages can't manually review all of them every month. This project builds a decision-support model that ranks pages by their probability of being in decline, so review effort goes to the pages that need it most.

## Dataset

- **Source:** [FlyRank/internship-warehouse](https://huggingface.co/datasets/FlyRank/internship-warehouse) (Hugging Face, gated full release) — this project uses the public, ungated **anonymized starter slice** from [flyrank-bih/flyrank-ml-internship-starter](https://github.com/flyrank-bih/flyrank-ml-internship-starter)
- **Size:** 30,000 rows × 44 columns, 32 pseudonymized clients
- **Target:** `is_declining_label` — derived from `trend_direction == "down"` (54.2% positive rate)

## Results (test set, client-holdout split)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 58.7% | 57.7% | 72.4% | 64.2% | 0.616 |
| Decision Tree | 56.5% | 55.8% | 71.5% | 62.7% | 0.592 |
| Random Forest | 58.4% | 58.1% | 66.7% | 62.1% | 0.609 |

Logistic Regression was selected: simplest model, best F1/AUC, fully interpretable coefficients. All three models land in a similar, modest range — this is an honest reflection of how hard the underlying signal is, not a tuning gap.

## Project structure

```
project/
├── data/
│   ├── raw/                          # content_refresh_anonymized.csv (not committed if large — see DATA_USE)
│   └── processed/
├── notebooks/
│   └── content_decline_analysis.ipynb  # full EDA -> model -> evaluation walkthrough
├── src/
│   ├── preprocessing.py              # cleaning + feature engineering
│   ├── train.py                      # trains + saves the pipeline
│   └── predict.py                    # scores new rows with a saved model
├── app/
│   └── app.py                        # Streamlit decline-risk demo
├── models/                           # saved model.pkl (gitignored)
├── outputs/
│   └── charts/                       # generated evaluation charts
├── requirements.txt
└── README.md
```

## Reproduce it

```bash
git clone <your-repo-url>
cd content-decline-prediction
pip install -r requirements.txt

# train
python src/train.py --input data/raw/content_refresh_anonymized.csv --output models/model.pkl

# score new rows
python src/predict.py --model models/model.pkl --input data/raw/some_new_export.csv

# run the notebook
jupyter notebook notebooks/content_decline_analysis.ipynb

# run the demo app locally
streamlit run app/app.py
```

## Limitations

- Trained on a 30k-row teaching slice of a much larger (~79M-row) warehouse — patterns here are directional, not the full population.
- AUC ≈ 0.6 means this is a ranking/triage signal, not a reliable per-page verdict.
- 32 clients contribute unevenly-sized samples; some patterns (e.g. content-type effects) may be client-driven rather than universal.
- Associations are correlational. Nothing here should be read as "if you update this page, it will stop declining."

## Data use

This project follows the FlyRank internship's data-use terms: only the anonymized starter slice is used, no client-identifying information is included anywhere in this repo or the accompanying write-up, and all results are framed as observed/directional rather than causal.
