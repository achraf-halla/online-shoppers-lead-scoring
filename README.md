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
│   └── 01_eda.ipynb      # exploratory analysis
├── src/
│   ├── data.py           # dataset download + load helpers
│   └── features.py       # feature engineering (Phase 2)
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
- [ ] Baseline logistic regression with class weighting
- [ ] Gradient boosting (LightGBM) with stratified CV
- [ ] Probability calibration (Platt / isotonic) and reliability diagrams
- [ ] SHAP-based feature attribution and per-prediction explanations
- [ ] Top-decile precision and lift analysis (the metrics a marketing team
      actually cares about)

## Results

*To be filled in after Phase 2.*

## License

MIT — see [LICENSE](LICENSE).
