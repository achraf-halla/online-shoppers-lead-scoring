"""Download and load the Online Shoppers Purchasing Intention dataset.

Run as a script to prefetch the data:

    python src/data.py

Or import the helpers from a notebook:

    from src.data import load

    df = load()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / "online_shoppers_intention.csv"

UCI_DATASET_ID = 468  # Online Shoppers Purchasing Intention Dataset


def fetch() -> pd.DataFrame:
    """Download the dataset from the UCI ML Repository."""
    from ucimlrepo import fetch_ucirepo

    print(f"Fetching UCI dataset id={UCI_DATASET_ID}...")
    ds = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols to {RAW_PATH}")
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    """Load the dataset from local cache, fetching it if missing."""
    if refresh or not RAW_PATH.exists():
        return fetch()
    return pd.read_csv(RAW_PATH)


if __name__ == "__main__":
    df = load(refresh=True)
    print("\nShape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nTarget distribution:")
    print(df["Revenue"].value_counts(normalize=True).round(4))
