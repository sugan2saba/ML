# scripts/promote_model.py
#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

from mlflow.tracking import MlflowClient


def die(msg: str, code: int = 1):
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(code)


def get_tracking_uri(cli_uri: Optional[str]) -> str:
    if cli_uri:
        return cli_uri
    env = os.getenv("MLFLOW_TRACKING_URI")
    if env:
        return env
    # default MLflow local file store (fine for testing, but no registry server)
    return "./mlruns"


def pick_latest_version(client: MlflowClient, model_name: str) -> str:
    """
    Returns the numerically largest version string for the given registered model.
    Uses search_model_versions for robustness.
    """
    pages = client.search_model_versions(filter_string=f"name='{model_name}'")
    if not pages:
        die(f"No versions found for registered model '{model_name}'.")
    try:
        latest = max(pages, key=lambda mv: int(mv.version))
    except Exception:
        # Fallback in case version cannot be cast (unlikely)
        latest = sorted(pages, key=lambda mv: mv.version)[-1]
    return latest.version


def ensure_model_exists(client: MlflowClient, model_name: str):
    try:
        client.get_registered_model(model_name)
    except Exception:
        die(f"Registered model '{model_name}' not found. Check the name or create it by logging a model.")


def set_alias(client: MlflowClient, model_name: str, alias: str, version: str):
    """
    Preferred modern approach: map alias to version. Re-pointing the alias
    automatically moves it from any previous version to the new one.
    """
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    print(f"[ok] Set alias '{alias}' -> {model_name} v{version}")
    print(f"     Load with: mlflow.pyfunc.load_model('models:/{model_name}@{alias}')")


def maybe_transition_stage(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: Optional[str],
    archive_existing: bool,
):
    """
    Backward-compat: only call this if the user explicitly asks for --stage.
    NOTE: This uses a deprecated API and may be removed in a future MLflow release.
    """
    if not stage:
        return

    # Warn clearly, but still do it because user asked for it.
    print(
        "[warn] Using model stages (deprecated in MLflow). "
        "Prefer --alias instead. See: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages"
    )
    # Keep import local to reduce static import warnings in newer versions.
    from mlflow.tracking.client import MlflowClient as _DeprecatedClient  # same class, just clarifies the warning source
    _DeprecatedClient().transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    print(f"[ok] Transitioned stage: {model_name} v{version} -> {stage}")


def main():
    parser = argparse.ArgumentParser(
        description="Promote a registered MLflow model version via Alias (recommended) or Stage (deprecated)."
    )
    parser.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", "MediWatchReadmit"),
                        help="Registered Model name (default: env MLFLOW_MODEL_NAME or 'MediWatchReadmit').")
    parser.add_argument("--version", type=int, default=None,
                        help="Specific version to promote. If omitted, latest version is selected automatically.")
    parser.add_argument("--alias", default=None,
                        help="Alias to set on the chosen version (e.g., Production, Staging, Canary). Recommended.")
    parser.add_argument("--stage", choices=["Staging", "Production"], default=None,
                        help="(Deprecated) Stage to transition to. Prefer --alias.")
    parser.add_argument("--no-archive-existing", action="store_true",
                        help="When using --stage, do not archive existing versions in that stage.")
    parser.add_argument("--tracking-uri", default=None,
                        help="Override MLflow tracking URI (else env MLFLOW_TRACKING_URI or ./mlruns).")

    args = parser.parse_args()

    tracking_uri = get_tracking_uri(args.tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri  # ensure client uses it
    print(f"[info] Tracking URI: {tracking_uri}")

    client = MlflowClient(tracking_uri)
    model_name = args.model_name
    ensure_model_exists(client, model_name)

    # Choose version (auto-latest if not provided)
    if args.version is None:
        version = pick_latest_version(client, model_name)
        print(f"[info] Auto-selected latest version: v{version}")
    else:
        version = str(args.version)

    # Preferred: set alias if provided
    if args.alias:
        set_alias(client, model_name, args.alias, version)

    # Backward-compat: set stage only if explicitly requested
    if args.stage:
        maybe_transition_stage(
            client,
            model_name=model_name,
            version=version,
            stage=args.stage,
            archive_existing=not args.no_archive_existing,
        )

    if not args.alias and not args.stage:
        print("[note] No --alias or --stage provided. Nothing to promote. "
              "Use --alias (recommended) or --stage (deprecated).")


if __name__ == "__main__":
    main()
