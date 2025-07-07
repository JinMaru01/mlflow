import os
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# --- S3 / MinIO Config ---
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")     
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL") 

# --- MLflow Config ---
TRACKING_URI = os.getenv("TRACKING_URI")
EXPERIMENT_NAME = "KMeans Clustering"
BASE_RUN_NAME = "KMeans Clustering"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(TRACKING_URI)

# Get or create experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# Determine next version number
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=100,
)

max_version = 0
for run in runs:
    run_name = run.data.tags.get("mlflow.runName", "")
    if run_name.startswith(BASE_RUN_NAME) and " v" in run_name:
        try:
            version_num = int(run_name.split(" v")[-1])
            max_version = max(max_version, version_num)
        except ValueError:
            pass

current_version = max_version + 1
current_run_name = f"{BASE_RUN_NAME} v{current_version}"
print(f"🚀 Starting run: {current_run_name}")

mlflow.sklearn.autolog()

# --- Load data and prepare for clustering ---
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_true = iris.target  # Ground truth for evaluation (not used in training)

# Scale features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with mlflow.start_run(experiment_id=experiment_id, run_name=current_run_name) as run:
    # Hyperparameters
    n_clusters = 3
    random_state = 42
    max_iter = 300
    
    mlflow.log_param("version", current_version)
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("scaled_features", True)

    # Train KMeans model
    model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
    cluster_labels = model.fit_predict(X_scaled)

    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    inertia = model.inertia_
    
    # If ground truth is available, calculate external validation metrics
    ari_score = adjusted_rand_score(y_true, cluster_labels)
    nmi_score = normalized_mutual_info_score(y_true, cluster_labels)

    # Log metrics
    mlflow.log_metrics({
        "silhouette_score": silhouette_avg,
        "inertia": inertia,
        "adjusted_rand_score": ari_score,
        "normalized_mutual_info_score": nmi_score
    })

    # Create visualizations
    # 1. Cluster scatter plot (using first two features for visualization)
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('KMeans Clustering Results')
    plt.legend()
    plt.colorbar(scatter)
    
    # Plot 2: Ground truth (for comparison)
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('Ground Truth Classes')
    plt.colorbar(scatter2)
    
    plt.tight_layout()
    plt.savefig("clustering_results.png", dpi=300, bbox_inches='tight')
    mlflow.log_artifact("clustering_results.png")
    plt.close()

    # 2. Elbow method visualization (for optimal k selection)
    k_range = range(1, 11)
    inertias = []
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter)
        kmeans_temp.fit(X_scaled)
        inertias.append(kmeans_temp.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.axvline(x=n_clusters, color='red', linestyle='--', label=f'Selected k={n_clusters}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("elbow_method.png", dpi=300, bbox_inches='tight')
    mlflow.log_artifact("elbow_method.png")
    plt.close()

    # 3. Silhouette analysis
    from sklearn.metrics import silhouette_samples
    
    silhouette_scores = silhouette_samples(X_scaled, cluster_labels)
    
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(n_clusters):
        cluster_silhouette_scores = silhouette_scores[cluster_labels == i]
        cluster_silhouette_scores.sort()
        
        size_cluster_i = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_scores,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    plt.xlabel('Silhouette coefficient values')
    plt.ylabel('Cluster label')
    plt.title('Silhouette Analysis')
    plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
                label=f'Average Score: {silhouette_avg:.3f}')
    plt.legend()
    plt.savefig("silhouette_analysis.png", dpi=300, bbox_inches='tight')
    mlflow.log_artifact("silhouette_analysis.png")
    plt.close()

    # Log additional data
    cluster_summary = pd.DataFrame({
        'cluster': range(n_clusters),
        'size': [np.sum(cluster_labels == i) for i in range(n_clusters)]
    })
    cluster_summary.to_csv("cluster_summary.csv", index=False)
    mlflow.log_artifact("cluster_summary.csv")

    # Log model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log scaler as well (important for inference)
    mlflow.sklearn.log_model(scaler, artifact_path="scaler")

    # Register model to model registry
    model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name="IrisKMeansClassifier")

    print(f"✅ Model logged and registered: {result.name} v{result.version}")
    print(f"📊 Silhouette Score: {silhouette_avg:.3f}")
    print(f"📊 Inertia: {inertia:.3f}")
    print(f"📊 Adjusted Rand Score: {ari_score:.3f}")
    print(f"📊 Normalized Mutual Info Score: {nmi_score:.3f}")

    # Print cluster summary
    print("\n📋 Cluster Summary:")
    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        print(f"   Cluster {i}: {cluster_size} samples")