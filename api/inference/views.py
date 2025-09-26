import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictRequestSerializer
from .loader import get_model, make_dataframe, get_expected_columns

THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

class HealthView(APIView):
    authentication_classes = []; permission_classes = []
    def get(self, request):
        cols = get_expected_columns()
        return Response({
            "status":"ok",
            "model_loaded": True if cols else False,
            "threshold": THRESHOLD,
            "expected_columns": cols,
        })

class SchemaView(APIView):
    authentication_classes = []; permission_classes = []
    def get(self, request):
        return Response({"expected_columns": get_expected_columns()})

class PredictView(APIView):
    authentication_classes = []; permission_classes = []

    def post(self, request):
        s = PredictRequestSerializer(data=request.data)
        if not s.is_valid():
            return Response({"error": s.errors}, status=status.HTTP_400_BAD_REQUEST)

        records = s.validated_data["records"]
        if not records:
            return Response({"error":"records cannot be empty"}, status=400)

        pipe = get_model()
        X = make_dataframe(records)

        try:
            probs = pipe.predict_proba(X)[:, 1]
        except Exception:
            # Some classifiers don't have predict_proba; fallback
            # (not expected for our HGB/LR/RF, but just in case)
            probs = pipe.decision_function(X)
            # squash to (0,1)
            import numpy as np
            probs = 1 / (1 + np.exp(-probs))

        preds = (probs >= THRESHOLD).astype(int).tolist()
        return Response({
            "threshold": THRESHOLD,
            "items": [{"prob": float(p), "pred": int(y)} for p, y in zip(probs, preds)]
        })
