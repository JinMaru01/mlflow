import os
import mlflow
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- MinIO / S3 Configuration ---
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")     
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL") 

# --- MLflow Tracking Configuration ---
TRACKING_URI = os.getenv("TRACKING_URI")
RUN_ID = "917ac26ed71b44969d7422441efeef3b"  # Your KMeans run ID
MODEL_NAME_IN_RUN = "model"  # The artifact path used in log_model()

# --- Set MLflow Tracking URI ---
mlflow.set_tracking_uri(TRACKING_URI)

# --- Build model URI ---
model_uri = f"runs:/{RUN_ID}/{MODEL_NAME_IN_RUN}"
print(f"📦 Loading KMeans model from: {model_uri}")

# --- Load model ---
try:
    # Load the KMeans model
    kmeans_model = mlflow.sklearn.load_model(model_uri)
    print("✅ KMeans model loaded successfully")
    
    # Try to load scaler from different possible locations
    scaler = None
    scaler_loaded = False
    
    # Method 1: Try loading scaler as sklearn model
    try:
        scaler_uri = f"runs:/{RUN_ID}/scaler"
        scaler = mlflow.sklearn.load_model(scaler_uri)
        print("✅ Scaler loaded successfully (sklearn format)")
        scaler_loaded = True
    except Exception as e:
        print(f"⚠️  Could not load scaler as sklearn model: {e}")
    
    # Method 2: Try loading scaler as pickle artifact
    if not scaler_loaded:
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(TRACKING_URI)
            
            # Download scaler.pkl artifact
            artifact_path = client.download_artifacts(RUN_ID, "scaler.pkl")
            with open(artifact_path, "rb") as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded successfully (pickle format)")
            scaler_loaded = True
        except Exception as e:
            print(f"⚠️  Could not load scaler as pickle: {e}")
    
    # Method 3: Create new scaler and fit on sample data (fallback)
    if not scaler_loaded:
        print("⚠️  No saved scaler found. Creating new StandardScaler...")
        print("   Note: This may affect prediction accuracy!")
        scaler = StandardScaler()
        # We'll fit it on the input data itself as a fallback
        scaler_loaded = True
    
    # Display model information
    print(f"\n📋 Model Info:")
    print(f"   - Number of clusters: {kmeans_model.n_clusters}")
    print(f"   - Cluster centers shape: {kmeans_model.cluster_centers_.shape}")
    print(f"   - Inertia: {kmeans_model.inertia_:.3f}")
    
    # --- Example input data (same format as training data) ---
    input_data = pd.DataFrame([
        {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        },
        {
            "sepal length (cm)": 6.7,
            "sepal width (cm)": 3.0,
            "petal length (cm)": 5.2,
            "petal width (cm)": 2.3
        },
        {
            "sepal length (cm)": 4.9,
            "sepal width (cm)": 3.1,
            "petal length (cm)": 1.5,
            "petal width (cm)": 0.2
        }
    ])
    
    print(f"\n📊 Input data:")
    print(input_data)
    
    # --- Preprocess input data (scale features) ---
    if scaler_loaded:
        # If scaler wasn't properly saved, fit it on input data as fallback
        if not hasattr(scaler, 'scale_'):
            print("⚠️  Fitting scaler on input data (fallback method)")
            scaler.fit(input_data)
        
        input_scaled = scaler.transform(input_data)
        print(f"\n📊 Scaled input data:")
        print(pd.DataFrame(input_scaled, columns=input_data.columns))
    else:
        # If no scaler available, use original data (not recommended)
        print("⚠️  No scaler available, using original data")
        input_scaled = input_data.values
    
    # --- Make predictions (cluster assignments) ---
    cluster_predictions = kmeans_model.predict(input_scaled)
    print(f"\n✅ Cluster Predictions:")
    print(f"   Clusters: {cluster_predictions}")
    
    # --- Calculate distances to cluster centers ---
    distances = kmeans_model.transform(input_scaled)
    print(f"\n📏 Distances to each cluster center:")
    for i, (row_idx, dist_row) in enumerate(zip(range(len(input_data)), distances)):
        print(f"   Sample {i+1}: {dist_row}")
        closest_cluster = np.argmin(dist_row)
        print(f"     -> Assigned to cluster {closest_cluster} (distance: {dist_row[closest_cluster]:.3f})")
    
    # --- Create results DataFrame ---
    results_df = pd.DataFrame({
        'sample_id': range(1, len(input_data) + 1),
        'sepal_length': input_data["sepal length (cm)"],
        'sepal_width': input_data["sepal width (cm)"],
        'petal_length': input_data["petal length (cm)"],
        'petal_width': input_data["petal width (cm)"],
        'predicted_cluster': cluster_predictions,
        'distance_to_assigned_cluster': [distances[i][cluster_predictions[i]] for i in range(len(cluster_predictions))]
    })
    
    print(f"\n📋 Results Summary:")
    print(results_df)
    
    # --- Display cluster centers ---
    print(f"\n🎯 Cluster Centers (in scaled space):")
    feature_names = input_data.columns
    centers_df = pd.DataFrame(kmeans_model.cluster_centers_, columns=feature_names)
    centers_df.index = [f"Cluster {i}" for i in range(kmeans_model.n_clusters)]
    print(centers_df)
    
    # --- Additional analysis ---
    print(f"\n📈 Additional Analysis:")
    for cluster_id in range(kmeans_model.n_clusters):
        samples_in_cluster = np.sum(cluster_predictions == cluster_id)
        print(f"   Cluster {cluster_id}: {samples_in_cluster} samples assigned")
    
    # --- Save results to CSV (optional) ---
    results_df.to_csv("kmeans_predictions.csv", index=False)
    print(f"\n💾 Results saved to 'kmeans_predictions.csv'")

except mlflow.exceptions.MlflowException as e:
    print("❌ MLflow error while loading model:")
    print(e)
    print("\n💡 Tips:")
    print("   - Check if the RUN_ID is correct")
    print("   - Ensure the model artifact exists in the run")
    print("   - Verify MLflow tracking URI is accessible")
except Exception as e:
    print("❌ Unexpected error:")
    print(e)


# --- Alternative: Load using pyfunc (unified interface) ---
print("\n" + "="*50)
print("🔄 Alternative: Using pyfunc interface")
print("="*50)

try:
    # Load model using pyfunc interface
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded via pyfunc interface")
    
    # Create input data for pyfunc test
    input_data_pyfunc = pd.DataFrame([
        {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    ])
    
    # Note: pyfunc interface might not handle preprocessing
    try:
        pyfunc_predictions = pyfunc_model.predict(input_data_pyfunc)
        print("✅ Pyfunc predictions:")
        print(pyfunc_predictions)
    except Exception as e:
        print("⚠️  Pyfunc prediction failed - likely needs preprocessing:")
        print(f"   Error: {e}")
        print("   💡 Use sklearn interface with manual scaling as shown above")

except mlflow.exceptions.MlflowException as e:
    print("❌ MLflow error with pyfunc:")
    print(e)