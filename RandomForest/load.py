import os
import mlflow
import pandas as pd

# --- MinIO / S3 Configuration ---
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")     
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL") 

# --- MLflow Tracking Configuration ---
TRACKING_URI = os.getenv("TRACKING_URI")
RUN_ID = "f33cc6b5591044129cd9720619483b0e"  # Your run ID
MODEL_NAME_IN_RUN = "model"  # The artifact path used in log_model()

# --- Set MLflow Tracking URI ---
mlflow.set_tracking_uri(TRACKING_URI)

# --- Build model URI ---
model_uri = f"runs:/{RUN_ID}/{MODEL_NAME_IN_RUN}"
print(f"📦 Loading model from: {model_uri}")

# --- Load and predict ---
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully")

    # --- Example input data ---
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
        }
    ])

    # --- Make prediction ---
    predictions = model.predict(input_data)
    print("✅ Predictions:")
    print(predictions)

except mlflow.exceptions.MlflowException as e:
    print("❌ MLflow error while loading model:")
    print(e)
except Exception as e:
    print("❌ Unexpected error:")
    print(e)
