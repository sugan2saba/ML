# scripts/make_splits.py
from pathlib import Path
import pandas as pd
from mediwatch.preprocess import load_raw, clean_and_engineer, split_train_valid_test

Path("/Users/suganthi/mediwatch/data/processed").mkdir(parents=True, exist_ok=True)
df = load_raw("/Users/suganthi/mediwatch/data/raw/diabetic_data.csv")
X, y, groups = clean_and_engineer(df)
Xtr, ytr, Xva, yva, Xte, yte = split_train_valid_test(X, y, groups)

# save as parquet for speed
Xtr.to_parquet("/Users/suganthi/mediwatch/data/processed/X_train.parquet")
ytr.to_frame("readmit_30d").to_parquet("/Users/suganthi/mediwatch/data/processed/y_train.parquet")
Xva.to_parquet("/Users/suganthi/mediwatch/data/processed/X_valid.parquet")
yva.to_frame("readmit_30d").to_parquet("/Users/suganthi/mediwatch/data/processed/y_valid.parquet")
Xte.to_parquet("/Users/suganthi/mediwatch/data/processed/X_test.parquet")
yte.to_frame("readmit_30d").to_parquet("/Users/suganthi/mediwatch/data/processed/y_test.parquet")

print("saved processed splits to data/processed/")
