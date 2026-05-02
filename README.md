# Online Shoppers Lead Scoring

Predicting purchase intent from e-commerce session behaviour, with a focus on
**interpretability** and **calibrated probabilities** — the things that
actually matter when a model feeds a marketing or sales workflow.

## Why this project

Lead scoring is a recurring problem in B2B and B2C: given a stream of sessions
or contacts, rank them by likelihood to convert so that limited human attention
goes to the right ones. A good model has to be:

1. **Calibrated** — scores should be usable as probabilities, not just rankings.
2. **Explainable** — sales/marketing colleagues need to understand *why* a
   contact ranks where it does.
3. **Useful at the threshold that matters** — top-decile precision is often
   more important than overall accuracy.

This repo walks through the full pipeline on a public dataset so the modelling
choices are inspectable, not magical.

## Dataset

[Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(UCI ML Repository, Sakar et al. 2018) — 12,330 sessions on a Turkish
e-commerce site over a one-year period. 17 features (10 numerical, 7
categorical) describe page visits, session duration, Google Analytics metrics
(BounceRate, ExitRate, PageValue), and visit context (month, weekend, visitor
type). Target: `Revenue` — whether the session ended in a purchase
(~15.5% positive class).

The dataset is not committed to this repo. Run `python src/data.py` to fetch it.

## Repo layout

```
online-shoppers-lead-scoring/
├── data/
│   ├── raw/              # downloaded CSV (gitignored)
│   └── processed/        # cleaned/feature-engineered data (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb      # exploratory analysis
│   └── 02_modeling.ipynb # baseline + LightGBM + calibration + SHAP
├── src/
│   ├── data.py           # dataset download + load helpers
│   ├── features.py       # feature recipes + LR pipeline factory
│   └── evaluation.py     # metrics, calibration, lift / gains plots
├── reports/
│   └── figures/          # plots saved from notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/data.py                 # downloads dataset to data/raw/
jupyter lab notebooks/01_eda.ipynb
```

## Roadmap

- [x] Repo scaffold, dataset loader, EDA notebook
- [x] Baseline logistic regression with class weighting
- [x] Gradient boosting (LightGBM) with stratified CV
- [x] Probability calibration (isotonic) and reliability diagrams
- [x] SHAP-based feature attribution and per-prediction explanations
- [x] Top-decile precision and cumulative gains analysis
- [ ] Hyperparameter search (Optuna) — left as a follow-up
- [ ] Per-segment evaluation (e.g. by `VisitorType`, `Month`)

## Approach

Two parallel models, each trained twice — with and without `PageValues`,
the feature most likely to leak conversion signal:

| Model | Preprocessing | Why |
| --- | --- | --- |
| Logistic regression | StandardScaler + OneHotEncoder, `class_weight="balanced"` | Linear, interpretable lower bound |
| LightGBM | Native categoricals, `is_unbalance=True`, isotonic calibration on a held-out validation slice | Handles mixed types and correlated features cleanly; SHAP-explainable |

The held-out test set is touched once at the end. CV happens on train.

## Results

*To be filled in after running `notebooks/02_modeling.ipynb` locally.*
The results section should at minimum cover:

- PR-AUC and ROC-AUC for both models, with and without PageValues.
- Top-10% precision (the "if sales calls the top decile" number).
- Brier score before and after isotonic calibration.
- The three highest-impact features per SHAP summary.

## License

MIT — see [LICENSE](LICENSE).
