#!/usr/bin/env sh
set -eu

# MODEL_NAME can be:
#   - prefix like "HGB" → picks newest /models/HGB*.joblib (or HGB_*.joblib/HGB-*.joblib)
#   - exact filename ending in ".joblib" → /models/<thatfile>
MODEL_NAME="${MODEL_NAME:-model_pipeline.joblib}"

pick_latest_by_pattern () {
  pattern="$1"
  # newest first by mtime
  found="$(find /models -maxdepth 1 -type f -name "$pattern" -printf '%T@ %p\n' 2>/dev/null \
           | sort -nr | head -n1 | awk '{print $2}')"
  if [ -n "${found:-}" ] && [ -f "$found" ]; then
    echo "$found"
  else
    echo ""
  fi
}

resolve_baked_model () {
  name="$1"
  # return absolute path or empty string
  case "$name" in
    *.joblib)
      [ -f "$name" ] && { echo "$name"; return; }
      [ -f "/models/$name" ] && { echo "/models/$name"; return; }
      echo ""
      ;;
    *)
      # try prefix patterns
      p=""
      p="${p:-$(pick_latest_by_pattern "${name}*.joblib")}"
      p="${p:-$(pick_latest_by_pattern "${name}_*.joblib")}"
      p="${p:-$(pick_latest_by_pattern "${name}-*.joblib")}"
      echo "$p"
      ;;
  esac
}

mkdir -p /models

chosen="$(resolve_baked_model "$MODEL_NAME")"
if [ -n "$chosen" ]; then
  # BAKED-IN MODE
  rm -f /models/current.joblib
  ln -s "$chosen" /models/current.joblib
  export MODEL_PATH="/models/current.joblib"
  echo "✅ Baked-in mode: using $chosen"
else
  # MLFLOW MODE
  # App will detect no MODEL_PATH file and load from MLflow in code.
  export MODEL_MODE="mlflow"
  echo "ℹ️ No baked model matched MODEL_NAME='$MODEL_NAME'. Falling back to MLflow mode."
  echo "   Ensure MLFLOW_TRACKING_URI (+ creds) and MODEL_NAME + MODEL_STAGE or MODEL_VERSION are set."
fi

exec "$@"
