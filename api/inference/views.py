# api/inference/views.py
from __future__ import annotations

import os
import platform
import socket
from typing import Any, Dict

import numpy as np
import pandas as pd
from django.utils.timezone import now
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response

from mediwatch_api.model_loader import model_info
from .loader import get_model, make_dataframe, get_expected_columns


class HealthView(APIView):
    """
    GET /health
    Lightweight health + model metadata. DRF view so it shows browsable UI in DEBUG.
    """
    def get(self, request):
        info = model_info()  # {source, path, name, stage, version, tracking_uri, mode_hint}
        data = {
            "status": "ok",
            "time": now().isoformat(),
            "service": "mediwatch-api",
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "debug": os.getenv("DEBUG", "0"),
            "model": info,
        }
        return Response(data)


@method_decorator(csrf_exempt, name="dispatch")
class PredictView(APIView):
    """
    POST /predict
    Body JSON: { "records": [ {<feature>: <value>, ...}, ... ] }

    - Uses make_dataframe() to align inputs to the training schema.
    - Calls the model's .predict().
    - If .predict_proba() exists, returns it as "probabilities".
    """
    def post(self, request):
        payload: Dict[str, Any] = request.data or {}
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            return Response({"error": "Expected 'records' as a non-empty list."}, status=400)

        try:
            df = make_dataframe(records)
        except Exception as e:
            return Response({"error": f"Failed to build dataframe: {e}"}, status=400)

        try:
            model = get_model()
        except Exception as e:
            return Response({"error": f"Failed to load model: {e}"}, status=500)

        try:
            y_pred = model.predict(df)
            preds = np.asarray(y_pred).tolist()
        except Exception as e:
            return Response({"error": f"Prediction failed: {e}"}, status=500)

        result: Dict[str, Any] = {
            "predictions": preds,
            "n": len(df),
            "columns": list(df.columns),
        }

        # Optional probabilities
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                result["probabilities"] = np.asarray(proba).tolist()
            except Exception:
                # Ignore probability errors silently to keep endpoint robust
                pass

        return Response(result)


class SchemaView(APIView):
    """
    GET /schema
    Returns the expected raw input column names as seen by the training pipeline.
    """
    def get(self, request):
        try:
            cols = get_expected_columns()
        except Exception as e:
            return Response({"error": f"Failed to retrieve schema: {e}"}, status=400)

        return Response({
            "expected_columns": cols,
            "count": len(cols),
        })
