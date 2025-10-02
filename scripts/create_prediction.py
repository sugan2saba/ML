# scripts/create_prediction.py
"""
Create or append prediction logs for Evidently drift monitoring.

Writes CSV at logs/predictions.csv by default with columns:
  - feature columns (Iris dataset): sepal_length, sepal_width, petal_length, petal_width
  - y_true (ground truth)
  - y_pred (predicted class)
  - proba_0, proba_1, proba_2 (class probabilities)
  - timestamp (unix seconds)  -> "now" by default, or --days-ago <N>

Examples:
  python scripts/create_prediction.py
  python scripts/create_prediction.py --n-rows 200 --mode overwrite
  python scripts/create_prediction.py --n-rows 60 --out logs/predictions.csv --mode append
  python scripts/create_prediction.py --n-rows 150 --mode overwrite --days-ago 7
"""

from __future__ import annotations
import argparse
import os
import pathlib
import time
from datetime import timedelta
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def build_dataset(
    test_size: float = 0.35,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, y_train, X_test, y_test


def train_and_predict(
    n_estimators: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    X_train, y_train, X_test, y_test = build_dataset(random_state=random_state)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)

    df = X_test.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]  # nice column names
    df["y_true"] = y_test.values
    df["y_pred"] = y_pred
    for i in range(proba.shape[1]):
        df[f"proba_{i}"] = proba[:, i]
    return df


def stamp_timestamp(df: pd.DataFrame, days_ago: int = 0) -> pd.DataFrame:
    """
    Add a unix 'timestamp' column. By default stamps 'now'.
    Use --days-ago N to stamp as if rows were created N days ago (for reference slices).
    """
    now = int(time.time())
    if days_ago and days_ago > 0:
        # subtract whole days in seconds
        now -= int(timedelta(days=days_ago).total_seconds())
    df = df.copy()
    df["timestamp"] = now
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate or append prediction logs for Evidently.")
    parser.add_argument("--out", default="logs/predictions.csv", help="Output CSV path (default: logs/predictions.csv)")
    parser.add_argument("--mode", choices=["append", "overwrite"], default="append",
                        help="Append to existing file or overwrite it (default: append)")
    parser.add_argument("--n-rows", type=int, default=80, help="How many rows to write (default: 80)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-estimators", type=int, default=100, help="RandomForest n_estimators (default: 100)")
    parser.add_argument("--days-ago", type=int, default=0,
                        help="Stamp rows as created N days ago (useful for 'reference' data). Default: 0 (now)")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a prediction DataFrame
    df = train_and_predict(n_estimators=args.n_estimators, random_state=args.random_state)

    # Keep exactly n-rows (sample if needed; if fewer available, use all)
    if args.n_rows < len(df):
        df = df.sample(n=args.n_rows, random_state=args.random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Stamp time (now or days-ago)
    df = stamp_timestamp(df, days_ago=args.days_ago)

    # Write out
    if args.mode == "overwrite" or not out_path.exists():
        df.to_csv(out_path, index=False)
        action = "Wrote"
    else:
        # Append without header
        df.to_csv(out_path, index=False, mode="a", header=False)
        action = "Appended"

    print(f"{action} {len(df)} rows -> {out_path.resolve()}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
