# scripts/monitor_evidently.py
# Generate an Evidently drift report from API prediction logs.
# Works locally and in CI (GitHub Actions).
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping


DEFAULT_FEATURES = [
    "race", "gender", "age",
    "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses",
    "diabetesMed", "A1Cresult", "max_glu_serum",
]

META_COLS = {"event_id", "ts", "model_version", "threshold"}
OUTPUT_COLS = {"prob", "pred"}


def load_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    df = pd.read_csv(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    return df


def choose_feature_cols(df: pd.DataFrame) -> List[str]:
    # Prefer known features if present, otherwise auto-infer (exclude meta/output cols)
    cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    if cols:
        return cols
    exclude = META_COLS.union(OUTPUT_COLS)
    return [c for c in df.columns if c not in exclude]


def slice_window(df: pd.DataFrame, since_epoch: int, until_epoch: int) -> pd.DataFrame:
    if "ts" not in df.columns:
        # no timestamps → use full DF (CI demos), but still copy() to avoid view issues
        return df.copy()
    since = pd.to_datetime(since_epoch, unit="s")
    until = pd.to_datetime(until_epoch, unit="s")
    return df[(df["ts"] >= since) & (df["ts"] < until)].copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-csv", default="logs/predictions.csv")
    p.add_argument("--out-dir", default="reports/monitoring")
    p.add_argument("--ref-window-sec", type=int, default=7*24*3600,
                   help="Reference window size in seconds (default: 7 days)")
    p.add_argument("--cur-window-sec", type=int, default=24*3600,
                   help="Current window size in seconds (default: 1 day)")
    p.add_argument("--reference-is-recent", action="store_true",
                   help="If set, both windows are the most recent; else reference is the window right before current.")
    p.add_argument("--min-rows", type=int, default=50,
                   help="Minimum rows required in each window to run Evidently.")
    p.add_argument("--target-column", default="pred",
                   help="Which output to treat as target for TargetDriftPreset: typically 'pred' or a true label.")
    p.add_argument("--fail-on-drift", action="store_true",
                   help="Exit with code 1 if Evidently flags dataset drift.")
    args = p.parse_args()

    log_path = Path(args.log_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(log_path)
    feat_cols = choose_feature_cols(df)

    now = int(time.time())
    if args.reference_is_recent:
        # Both windows end "now"; reference spans the last ref-window, current spans last cur-window
        ref_since = now - args.ref_window_sec
        ref_until = now
        cur_since = now - args.cur_window_sec
        cur_until = now
    else:
        # Non-overlapping: reference ends right before the current window
        ref_since = now - (args.ref_window_sec + args.cur_window_sec)
        ref_until = now - args.cur_window_sec
        cur_since = now - args.cur_window_sec
        cur_until = now

    ref_df = slice_window(df, ref_since, ref_until)
    cur_df = slice_window(df, cur_since, cur_until)

    # Keep only features + outputs if present
    keep = set(feat_cols) | OUTPUT_COLS
    ref_df = ref_df[[c for c in ref_df.columns if c in keep]].copy()
    cur_df = cur_df[[c for c in cur_df.columns if c in keep]].copy()

    # Basic sanity: enough rows?
    if len(ref_df) < args.min_rows or len(cur_df) < args.min_rows:
        msg = (f"[monitor] Not enough rows to run Evidently "
               f"(ref={len(ref_df)}, cur={len(cur_df)}, min={args.min_rows}). "
               f"Skipping report generation.")
        print(msg)
        # Still write a tiny JSON so CI steps don't fail unexpectedly.
        (out_dir / "evidently_report.json").write_text(json.dumps({
            "metrics": [],
            "note": msg,
            "ref_rows": len(ref_df),
            "cur_rows": len(cur_df),
        }, indent=2))
        return

    # Determine target column if present
    target_col = args.target_column if args.target_column in cur_df.columns else None

    # Build report (Evidently 0.4.x style)
    metrics = [DataDriftPreset()]
    if target_col:
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)

    if target_col:
        colmap = ColumnMapping(target=target_col)
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=colmap)
    else:
        report.run(reference_data=ref_df, current_data=cur_df)

    # Save outputs
    html_path = out_dir / "evidently_report.html"
    json_path = out_dir / "evidently_report.json"
    report.save_html(str(html_path))
    report.save_json(str(json_path))

    # Console summary + optional exit code
    summary = report.as_dict()
    data_drift = False
    drifted_cols = 0
    for m in summary.get("metrics", []):
        if m.get("metric") == "DataDriftPreset":
            res = m.get("result", {}) or {}
            data_drift = bool(res.get("dataset_drift", False))
            drifted_cols = int(res.get("number_of_drifted_columns", 0))
            break

    print(f"[monitor] dataset_drift={data_drift} drifted_columns={drifted_cols}")
    print(f"[monitor] saved: {html_path} and {json_path}")

    if args.fail_on_drift and data_drift:
        # Non-zero exit for CI gates
        raise SystemExit(1)


if __name__ == "__main__":
    main()
