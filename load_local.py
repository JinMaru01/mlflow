import mlflow

# Set local tracking URI if necessary
mlflow.set_tracking_uri("file:///C:/Users/darachin.kong/Desktop/Experiment/mlflow/mlruns")

# Load the model from the model registry by name and version
model = mlflow.pyfunc.load_model(model_uri="models:/IrisClassifier/3")

# Example: predict
import pandas as pd
input_data = pd.DataFrame([
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}
    ])
prediction = model.predict(input_data)
print(prediction)
