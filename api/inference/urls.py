from django.urls import path
from .views import HealthView, PredictView, SchemaView

urlpatterns = [
    path("health", HealthView.as_view(), name="health"),
    path("schema", SchemaView.as_view(), name="schema"),
    path("predict", PredictView.as_view(), name="predict"),
]
