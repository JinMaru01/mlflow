import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRACKING_URI = "http://10.120.210.56:5000"
EXPERIMENT_NAME = "Random Forest Classifier"
BASE_RUN_NAME = "Random Forest Classifier"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(TRACKING_URI)

# Get or create experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# Fetch recent runs and find max version from run names or params
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=100,
)

max_version = 0
for run in runs:
    run_name = run.data.tags.get("mlflow.runName", "")
    if run_name.startswith(BASE_RUN_NAME):
        # Try to parse version from run name suffix: e.g. "Random Forest Classifier v3"
        if " v" in run_name:
            try:
                version_num = int(run_name.split(" v")[-1])
                if version_num > max_version:
                    max_version = version_num
            except ValueError:
                pass

current_version = max_version + 1
current_run_name = f"{BASE_RUN_NAME} v{current_version}"

print(f"Starting run with name: {current_run_name}")

mlflow.sklearn.autolog()

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

with mlflow.start_run(experiment_id=experiment_id, run_name=current_run_name) as run:
    mlflow.log_param("version", current_version)
    mlflow.log_param("n_estimators", 80)

    model = RandomForestClassifier(n_estimators=80)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    print(f"Model and metrics logged with run name: {current_run_name}")
