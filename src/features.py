"""Feature preparation for the lead scoring models.

Two feature recipes share most of the work:

* Logistic regression — needs scaled numerics and one-hot-encoded
  categoricals; this is built as a `ColumnTransformer` so it composes
  cleanly into a `Pipeline`.
* LightGBM — handles categoricals natively, so we just cast them to
  pandas `category` dtype and pass `categorical_feature="auto"`.

The `drop_pagevalues` flag is the lever for the leakage-vs-realism
discussion: PageValues is computed partly from the conversion outcome,
so we benchmark with and without it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates",
    "PageValues", "SpecialDay",
]
CATEGORICAL_COLS = [
    "Month", "VisitorType", "Weekend",
    "OperatingSystems", "Browser", "Region", "TrafficType",
]
TARGET = "Revenue"


def prepare(df: pd.DataFrame, drop_pagevalues: bool = False
            ) -> Tuple[pd.DataFrame, pd.Series]:
    """Split frame into (X, y), cast categoricals, optionally drop PageValues."""
    df = df.copy()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    if drop_pagevalues and "PageValues" in X.columns:
        X = X.drop(columns=["PageValues"])

    return X, y


def make_split(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split that preserves class balance."""
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


def build_lr_pipeline(class_weight: str | dict | None = "balanced",
                      C: float = 1.0,
                      drop_pagevalues: bool = False) -> Pipeline:
    """Logistic regression pipeline: scale + one-hot + LR.

    Parameters
    ----------
    class_weight : balanced/None/dict — passed to LogisticRegression.
    C : inverse regularisation strength.
    drop_pagevalues : whether the numeric column list excludes PageValues.
    """
    num_cols = [c for c in NUMERICAL_COLS if not (drop_pagevalues and c == "PageValues")]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            C=C,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def lgbm_categorical_indices(X: pd.DataFrame) -> list[int]:
    """Column indices of categorical features, for LightGBM's categorical_feature."""
    return [i for i, c in enumerate(X.columns) if c in CATEGORICAL_COLS]
