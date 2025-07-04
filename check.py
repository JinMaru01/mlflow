from mlflow.tracking import MlflowClient

client = MlflowClient()
for mv in client.search_model_versions("name='IrisClassifier'"):
    print(dict(mv))

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="IrisClassifier",
    version=2,
    stage="Production"
)
