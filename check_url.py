import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI (remote or local MLflow server)
mlflow.set_tracking_uri("http://10.120.210.56:5000")

# Define the experiment name or ID
EXPERIMENT_NAME = "Random Forest Classifier"
client = MlflowClient()

# Get experiment details
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"Experiment '{EXPERIMENT_NAME}' not found.")
else:
    print(f"Runs for experiment: {EXPERIMENT_NAME} (ID: {experiment.experiment_id})")

    # Search and list all runs
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "[Unnamed]")
        print(f"- Run ID: {run.info.run_id}, Name: {run_name}")
