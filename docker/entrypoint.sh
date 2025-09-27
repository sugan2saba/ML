#!/usr/bin/env sh
set -eu

# You can pass either:
#  - MODEL_NAME=HGB             (prefix: pick newest /models/HGB*.joblib)
#  - MODEL_NAME=HGB_*.joblib    (explicit glob)
#  - MODEL_NAME=somefile.joblib (exact filename)
MODEL_NAME="${MODEL_NAME:-model_pipeline.joblib}"

pick_latest_by_pattern () {
  pattern="$1"
  # Find all matching files and sort by mtime (newest first), pick first
  # NOTE: assumes no spaces in paths. If you have spaces, tighten this with -printf and -z sort.
  found="$(find /models -maxdepth 1 -type f -name "$pattern" -printf '%T@ %p\n' \
           | sort -nr \
           | head -n1 \
           | awk '{print $2}')"
  if [ -n "${found:-}" ] && [ -f "$found" ]; then
    echo "$found"
  else
    echo ""
  fi
}

resolve_model_path () {
  name="$1"
  case "$name" in
    *.joblib)
      # Treat as exact filename (allow absolute or just basename)
      if [ -f "$name" ]; then
        echo "$name"
        return 0
      elif [ -f "/models/$name" ]; then
        echo "/models/$name"
        return 0
      else
        echo ""
        return 0
      fi
      ;;
    *)
      # Treat as prefix – try common patterns (broadest first)
      p=""
      p="${p:-$(pick_latest_by_pattern "${name}*.joblib")}"
      p="${p:-$(pick_latest_by_pattern "${name}_*.joblib")}"
      p="${p:-$(pick_latest_by_pattern "${name}-*.joblib")}"
      echo "$p"
      return 0
      ;;
  esac
}

chosen="$(resolve_model_path "$MODEL_NAME")"

if [ -z "$chosen" ]; then
  echo "❌ Could not resolve model for MODEL_NAME='$MODEL_NAME'"
  echo "   Available models:"
  ls -1 /models/*.joblib 2>/dev/null || echo "   (none found)"
  exit 1
fi

# Create stable symlink
rm -f /models/current.joblib
ln -s "$chosen" /models/current.joblib

echo "✅ Using model: $chosen"
export MODEL_PATH="/models/current.joblib"

# Hand off to the main process (Gunicorn CMD below)
exec "$@"
