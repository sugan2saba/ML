# scripts/feature_selection_probe.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.inspection import permutation_importance

from mediwatch.preprocess import build_preprocessor

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("/Users/suganthi/mediwatch/data/processed")
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("artifacts"); ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
Xtr = pd.read_parquet(DATA_DIR / "X_train.parquet")
ytr = pd.read_parquet(DATA_DIR / "y_train.parquet")["readmit_30d"].astype(int)
Xva = pd.read_parquet(DATA_DIR / "X_valid.parquet")
yva = pd.read_parquet(DATA_DIR / "y_valid.parquet")["readmit_30d"].astype(int)

# -----------------------------
# Preprocess + Model
# -----------------------------
pre, num_cols, cat_cols = build_preprocessor(Xtr)

model = HGB(
    learning_rate=0.05,
    max_iter=500,
    random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", model)])

# -----------------------------
# Fit
# -----------------------------
pipe.fit(Xtr, ytr)

# -----------------------------
# Predict
# -----------------------------
y_prob = pipe.predict_proba(Xva)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# -----------------------------
# Metrics
# -----------------------------
metrics = {
    "backend": "sklearn.HistGradientBoosting",
    "auroc": float(roc_auc_score(yva, y_prob)),
    "prauc": float(average_precision_score(yva, y_prob)),
    "brier": float(brier_score_loss(yva, y_prob)),
    "pos_rate_valid": float(yva.mean())
}

print("\n=== Validation Metrics ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k:>15}: {v:.4f}")
    else:
        print(f"{k:>15}: {v}")

with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# -----------------------------
# Curves
# -----------------------------
fpr, tpr, _ = roc_curve(yva, y_prob)
prec, rec, _ = precision_recall_curve(yva, y_prob)

# -----------------------------
# Feature Importance (Permutation)
# -----------------------------
Xva_t = pipe.named_steps["pre"].transform(Xva)
feat_names = pipe.named_steps["pre"].get_feature_names_out()
clf = pipe.named_steps["clf"]

r = permutation_importance(
    clf, Xva_t, yva,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc"
)

fi = pd.DataFrame({
    "feature": feat_names,
    "importance_mean": r.importances_mean,
    "importance_std": r.importances_std
}).sort_values("importance_mean", ascending=False)

fi.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
top = fi.head(25)[::-1]

# -----------------------------
# Composite Figure
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MediWatch — Feature Utility Probe", fontsize=16, fontweight="bold")

# ROC Curve
axes[0, 0].plot(fpr, tpr, label=f"AUROC={metrics['auroc']:.3f}")
axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].set_title("ROC Curve")
axes[0, 0].legend()

# PR Curve
axes[0, 1].plot(rec, prec, label=f"PR-AUC={metrics['prauc']:.3f}")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision-Recall Curve")
axes[0, 1].legend()

# Feature importance
axes[1, 0].barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
axes[1, 0].set_xlabel("Permutation Importance (Δ AUROC)")
axes[1, 0].set_title("Top 25 Feature Importances")

# Metrics block
axes[1, 1].axis("off")
text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                  for k, v in metrics.items()])
axes[1, 1].text(0.0, 0.5, text, fontsize=12, va="center", ha="left")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(REPORTS_DIR / "feature_selection_probe.png", dpi=150)
plt.close()

# -----------------------------
# Save predictions & model
# -----------------------------
preds_df = pd.DataFrame({
    "row_id": np.arange(len(yva)),
    "y_true": yva.values,
    "y_prob": y_prob,
    "y_pred_0p5": y_pred
})
preds_df.to_csv(REPORTS_DIR / "valid_predictions.csv", index=False)

joblib.dump(pipe, ARTIFACTS_DIR / "model_pipeline.joblib")

print("\n=== Saved Artifacts ===")
print(f"- {REPORTS_DIR/'metrics.json'}")
print(f"- {REPORTS_DIR/'feature_importance.csv'}")
print(f"- {REPORTS_DIR/'valid_predictions.csv'}")
print(f"- {REPORTS_DIR/'feature_selection_probe.png'}  <-- main summary figure")
print(f"- {ARTIFACTS_DIR/'model_pipeline.joblib'}")
