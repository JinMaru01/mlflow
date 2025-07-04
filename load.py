import mlflow
import mlflow.pyfunc
import pandas as pd

# --- Set MLflow Tracking URI ---
mlflow.set_tracking_uri("http://10.120.210.56:5000")

# --- Use pyfunc to avoid registry dependency ---
model_uri = "runs:/7d344f95837a4dd7892bf6b9caa41ec3/model"
print(f"📦 Loading model from: {model_uri}")

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully")

    input_data = pd.DataFrame([
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}
    ])

    predictions = model.predict(input_data)
    print("✅ Predictions:", predictions)

except Exception as e:
    print("❌ Failed to load model:")
    print(e)
