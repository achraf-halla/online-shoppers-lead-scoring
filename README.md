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

Held-out test set: 2,466 sessions, 15.5% positive class.

| Model                          | PR-AUC | ROC-AUC | Brier | prec@top10% |
|--------------------------------|--------|---------|-------|-------------|
| Logistic regression (with PV)  | 0.622  | 0.893   | 0.124 | 0.707       |
| Logistic regression (no PV)    | 0.321  | 0.735   | 0.215 | 0.382       |
| **LightGBM (with PV)**         | **0.743** | **0.932** | **0.094** | **0.781** |
| LightGBM (no PV)               | 0.365  | 0.776   | 0.177 | 0.423       |

### Key findings

- **PageValues drives most of the apparent performance.** Removing it
  nearly halves PR-AUC for both models. PageValues is partly computed
  from the conversion outcome, so the headline ~0.93 ROC-AUC commonly
  reported on this dataset is largely a leakage artefact. Without it,
  LightGBM still produces useful rankings (PR-AUC 0.37, top-decile lift
  2.7×) — the gap quantifies the value of a feature that wouldn't be
  available at real-time scoring time.
- **Top decile.** Calling the top 10% of LightGBM-ranked sessions
  reaches 78% precision against a 15.5% base rate (lift of 5×) and
  captures half of all converters. The top 30% captures 89% of
  converters at 46% precision.
- **Calibration is a trade-off.** Isotonic regression improves Brier
  score 23% (0.094 → 0.072) but costs ~4 pts of top-10% precision.
  Use isotonic when downstream needs probabilities (expected revenue,
  confidence thresholds); use raw scores when downstream only needs a
  ranking.
- **Top drivers (SHAP, no-PageValues model).** ExitRates dominates
  (≈2× the next feature), followed by Month and ProductRelated_Duration.
  Behavioural features (rates, durations) outweigh context features
  (Browser, Region, OS) by an order of magnitude. Month picks up a clear
  November pre-holiday spike.

## License

MIT — see [LICENSE](LICENSE).
