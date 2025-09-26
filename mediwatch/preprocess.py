# mediwatch/preprocess.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

MISSING_TOKEN = "?"
ORDINAL_MAX_GLU = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
ORDINAL_A1C = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}  # per UCI description. :contentReference[oaicite:2]{index=2}

def age_to_midpoint(s: pd.Series) -> pd.Series:
    # converts strings like "[70-80)" to midpoint 75.0
    def mid(x):
        m = re.match(r"\[(\d+)-(\d+)\)", str(x))
        return (float(m.group(1))+float(m.group(2)))/2 if m else np.nan
    return s.map(mid).astype(float)

def icd9_bucket(code: str) -> str:
    # map ICD-9 code (string) to broad body-system bucket
    # Reference groupings commonly used with this dataset:
    # 390–459 & 785: circulatory; 460–519: respiratory; 520–579: digestive; 250: diabetes;
    # 800–999: injury; 710–739: musculoskeletal; 580–629: genitourinary; 140–239: neoplasms; else: other.
    if code is None or pd.isna(code): return "UNK"
    # strip 'V'/'E' prefixes
    try:
        c = float(re.match(r"[VE]?(\d+\.?\d*)", str(code)).group(1))
    except Exception:
        return "OTHER"
    if (390 <= c <= 459) or c == 785: return "circulatory"
    if 460 <= c <= 519: return "respiratory"
    if 520 <= c <= 579: return "digestive"
    if 250 <= c < 251:  return "diabetes"
    if 800 <= c <= 999: return "injury"
    if 710 <= c <= 739: return "musculoskeletal"
    if 580 <= c <= 629: return "genitourinary"
    if 140 <= c <= 239: return "neoplasms"
    return "other"

class ICD9Bucketizer(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].apply(icd9_bucket)
        return X

@dataclass
class PreprocessArtifacts:
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    num_cols: List[str]
    low_card_cats: List[str]
    high_card_cats: List[str]

def load_raw(path="data/raw/diabetic_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).replace(MISSING_TOKEN, np.nan)
    return df

def clean_and_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # 1) binary target
    y = (df["readmitted"] == "<30").astype(int)

    # 2) safe identifier for group split
    groups = df["patient_nbr"]  # do NOT use as a feature

    # 3) drop columns we won't use as predictors (IDs, duplicates)
    drop_cols = ["encounter_id", "patient_nbr"]
    df = df.drop(columns=drop_cols)

    # 4) normalize categories
    df["gender"] = df["gender"].fillna("Unknown").replace({"Unknown/Invalid":"Unknown"})
    df["age_mid"] = age_to_midpoint(df["age"])

    # 5) ordinal meds/labs
    df["max_glu_serum_ord"] = df["max_glu_serum"].map(ORDINAL_MAX_GLU).astype("float")
    df["A1Cresult_ord"] = df["A1Cresult"].map(ORDINAL_A1C).astype("float")

    # 6) ICD-9 buckets
    diag_cols = ["diag_1","diag_2","diag_3"]
    df = ICD9Bucketizer(diag_cols).fit_transform(df)

    # 7) convenience flags
    df["is_med_changed"] = (df["change"] == "Ch").astype(int)   # 'Ch' means med change in this dataset. :contentReference[oaicite:3]{index=3}
    df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

    # 8) candidate feature lists
    num_cols = [
        "time_in_hospital","num_lab_procedures","num_procedures","num_medications",
        "number_outpatient","number_emergency","number_inpatient","number_diagnoses",
        "age_mid","max_glu_serum_ord","A1Cresult_ord"
    ]
    low_card_cats = [
        "race","gender","admission_type_id","discharge_disposition_id","admission_source_id",
        "change","diabetesMed","readmitted"  # we will NOT use 'readmitted' as feature, it was used for label above
    ]
    # remove the accidental 'readmitted' from low_card_cats:
    low_card_cats = [c for c in low_card_cats if c != "readmitted"]

    # ‘medical_specialty’ is high-card; ICD buckets + specialty can be useful
    high_card_cats = ["medical_specialty"] + diag_cols

    # 9) drop very sparse columns (weight, payer_code are often ~97%+ missing in this dataset)
    for c in ["weight","payer_code"]:
        if c in df.columns: df = df.drop(columns=[c])

    # 10) return artifacts
    X = df.drop(columns=["readmitted","age"])  # remove raw target & raw age band
    return X, y, groups

def split_train_valid_test(X, y, groups, valid_size=0.2, test_size=0.2, random_state=42):
    # Group-aware: patients do not cross splits (avoid leakage)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_tr, idx_te = next(gss1.split(X, y, groups))
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]
    g_tr, g_te = groups.iloc[idx_tr], groups.iloc[idx_te]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_size/(1-test_size), random_state=random_state)
    idx_tr2, idx_va = next(gss2.split(X_tr, y_tr, g_tr))
    return (X_tr.iloc[idx_tr2], y_tr.iloc[idx_tr2],
            X_tr.iloc[idx_va], y_tr.iloc[idx_va],
            X_te, y_te)

def build_preprocessor(X: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    # infer basic column groups from dtypes + names
    num_cols = [c for c in X.columns if X[c].dtype.kind in "if" and c not in []]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # split cats into low vs high cardinality to keep OHE size reasonable
    low_card, high_card = [], []
    for c in cat_cols:
        n = X[c].nunique(dropna=True)
        (low_card if n <= 30 else high_card).append(c)

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat_low",
         Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                   ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         low_card),
        ("cat_high",
         Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                   ("freq", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False))]),
         high_card),
    ])
    pipe = Pipeline([("pre", pre)])  # model will be attached later
    return pipe, num_cols, low_card + high_card
