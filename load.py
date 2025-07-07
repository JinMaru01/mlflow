import mlflow
import pandas as pd

# --- Configuration ---
TRACKING_URI = "http://10.120.210.54:5000"
RUN_ID = "7d344f95837a4dd7892bf6b9caa41ec3"  # Replace with your actual run ID
MODEL_NAME_IN_RUN = "Random Forest Classifier"  # Default name used by sklearn.autolog()

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
