#pip install mlflow scikit-learn
import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Point to the MLflow server you started with docker compose
mlflow.set_tracking_uri("http://localhost:5000")

X, y = load_iris(return_X_y=True)
Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(Xtr, ytr)
    mlflow.log_param("n_estimators", 50)
    mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name="HGB")
    run_id = run.info.run_id
    print("Run:", run_id)

# Promote latest version of HGB to Production
client = MlflowClient()
latest = client.get_latest_versions(name="HGB", stages=["None"])
if latest:
    v = latest[0].version
    client.transition_model_version_stage(
        name="HGB",
        version=v,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted HGB v{v} to Production")
else:
    print("No new versions found to promote")
