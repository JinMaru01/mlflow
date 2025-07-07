import pandas as pd
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import psycopg2
import mlflow
import mlflow.keras


def preprocess_data(data, n_in=1, n_out=1, dropnan=True):
    """
    Preprocess the input time series data from PostgreSQL for model testing.

    Parameters:
    - data: Input DataFrame from PostgreSQL.
    - n_in: Number of lag observations as input (X).
    - n_out: Number of observations as output (y).
    - dropnan: Whether to drop rows with NaN values.

    Returns:
    - Preprocessed DataFrame ready for model testing.
    """
    # Interpolate missing values
    clean_data = data.interpolate(method='linear', axis=0, limit=10)

    # Drop unwanted column
    # clean_data = clean_data.drop(columns=['global_intensity'])

    # Resample data to hourly intervals
    resampled_data = clean_data.resample('H').mean()

    #Credit: Adopted from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        dff = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(dff.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(dff.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    # Feature engineering for LSTM
    reframed_data = series_to_supervised(resampled_data, 1, 1)
    print(reframed_data.head())

    # Drop unwanted columns
    # drop columns we don't want
    reframed_data.drop(reframed_data.columns[[7,8,9,10,11]], axis=1, inplace=True)
    print(reframed_data.columns)

    # Return values
    return reframed_data.values



import pandas as pd
from sqlalchemy import create_engine, inspect

# Function to dynamically select all columns except a specified one
def generate_query_excluding_column(db_url, table_name, exclude_column):
    """
    Generate a SQL query that selects all columns except the specified column.

    Parameters:
    - db_url: Database connection URL.
    - table_name: Name of the table to query.
    - exclude_column: Column to exclude from the SELECT statement.

    Returns:
    - A SQL query string.
    """
    # Create database engine
    engine = create_engine(db_url)

    # Fetch columns from the table
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]

    # Exclude the specified column
    selected_columns = [col for col in columns if col != exclude_column]

    # Generate the query
    query = f"SELECT {', '.join(selected_columns)} FROM {table_name};"
    return query

# Function to connect to PostgreSQL and load data
def load_data_from_postgres(db_url, table_name, exclude_column):
    """
    Load data from PostgreSQL.

    Parameters:
    - db_url: Database connection URL.
    - table_name: Name of the table to query.
    - exclude_column: Column to exclude from the SELECT statement.

    Returns:
    - DataFrame containing the data from the table.
    """
    # Create database engine
    engine = create_engine(db_url)

    # Generate the query excluding the specified column
    query = generate_query_excluding_column(db_url, table_name, exclude_column)

    # Query the table and load data into a DataFrame
    data = pd.read_sql(query, con=engine, index_col='dt')  # Assume 'dt' is the index

    return data

# Database connection URL
db_url = "postgresql://postgres:1234@100.102.50.3:5432/postgres"

# Load the data
table_name = "mlflow_test"  # Replace with your table name
exclude_column = "global_intensity"

data = load_data_from_postgres(db_url, table_name, exclude_column)

data.columns

# Preprocess the data using preprocess_data
preprocessed_data = preprocess_data(data)

# Now `preprocessed_data` is ready for model testing
print(preprocessed_data)



feature_cols = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)','var6(t-1)']
target_col = 'var1(t)'

train_index = 500*24 #The logic is to have 500 days worth of training data. this could also be a hyperparameter that can be tuned.
train = preprocessed_data[:train_index, :]
test = preprocessed_data[train_index:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].




def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to retrain the model
def retrain_model(X_train, y_train, X_test, y_test):
    model = create_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=20, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1, shuffle=False)

    # Evaluate the retrained model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return model, history, rmse, r2, mae

# Function to evaluate and retrain, and log with MLflow
def evaluate_and_retrain(X_train, y_train, X_test, y_test, model, baseline_metrics, thresholds):
    # Debug shapes of inputs
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Ensure X_test has correct dimensions
    if len(X_test.shape) != 3:
        print("Reshaping X_test for LSTM input.")
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Check for NaN or Inf values
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        raise ValueError("X_test contains NaN or Inf values!")
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
        raise ValueError("y_test contains NaN or Inf values!")

    with mlflow.start_run(nested=True):  # Use nested runs for retraining
        # Try to predict
        try:
            predictions = model.predict(X_test)
            print("Sample Predictions:", predictions[:5])  # Check a few predictions
        except Exception as e:
            print("Error during prediction:", str(e))
            raise e

        # Calculate evaluation metrics
        current_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        current_r2 = r2_score(y_test, predictions)
        current_mae = mean_absolute_error(y_test, predictions)

        print(f"Current RMSE: {current_rmse}, Current R2: {current_r2}, Current MAE: {current_mae}")

        # Retrain if performance drops
        if (
            current_rmse > baseline_metrics["rmse"] + thresholds["rmse"]
            or current_r2 < baseline_metrics["r2"] - thresholds["r2"]
            or current_mae > baseline_metrics["mae"] + thresholds["mae"]
        ):
            print("Performance has decreased! Retraining the model...")
            retrained_model, history, new_rmse, new_r2, new_mae = retrain_model(X_train, y_train, X_test, y_test)

            # Log retrained model and metrics
            mlflow.keras.log_model(retrained_model, "retrained_model")
            mlflow.log_metric("new_rmse", new_rmse)
            mlflow.log_metric("new_r2_score", new_r2)
            mlflow.log_metric("new_mae", new_mae)

            # Update baseline metrics
            baseline_metrics["rmse"] = new_rmse
            baseline_metrics["r2"] = new_r2
            baseline_metrics["mae"] = new_mae
            print(f"Retrained Model RMSE: {new_rmse}, R2: {new_r2}, MAE: {new_mae}")
            return retrained_model, history
        else:
            print("Performance is acceptable. No retraining required.")
            return model, None

# Baseline metrics and thresholds
baseline_metrics = {"rmse": 0.5, "r2": 0.8, "mae": 0.3}  # Replace with your baseline values
thresholds = {"rmse": 0.1, "r2": 0.05, "mae": 0.05}  # Allowable performance drop

# MLflow experiment
mlflow.set_experiment("Automated LSTM Training and Evaluation")

# Ensure train_X and train_y have the same number of samples
train_X, train_y = train_X[:train_y.shape[0]], train_y[:train_X.shape[0]]

# Train and log the initial model
with mlflow.start_run():
    run_name = "Retrain LSTM model"
    mlflow.set_tag("mlflow.runName", run_name)
    # Log parameters
    mlflow.log_param("features", feature_cols)
    mlflow.log_param("target", target_col)
    
    # Train the initial model
    initial_model = create_model((train_X.shape[1], train_X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = initial_model.fit(train_X, train_y, epochs=50, batch_size=20, validation_data=(test_X, test_y), callbacks=[early_stop], verbose=1, shuffle=False)

    # Log the initial model
    mlflow.keras.log_model(initial_model, "initial_model")

    # Predict and calculate metrics
    predictions = initial_model.predict(test_X)
    initial_rmse = np.sqrt(mean_squared_error(test_y, predictions))
    initial_r2 = r2_score(test_y, predictions)
    initial_mae = mean_absolute_error(test_y, predictions)

    print(f"Initial Model RMSE: {initial_rmse}, R2: {initial_r2}, MAE: {initial_mae}")

    # Log metrics
    mlflow.log_metric("initial_rmse", initial_rmse)
    mlflow.log_metric("initial_r2_score", initial_r2)
    mlflow.log_metric("initial_mae", initial_mae)

    # Evaluate and retrain if needed
    final_model, retrain_history = evaluate_and_retrain(
        train_X, train_y, test_X, test_y, initial_model, 
        baseline_metrics, thresholds
    )

    # Log final metrics if retraining occurred
    if retrain_history is not None:
        final_predictions = final_model.predict(test_X)
        final_rmse = np.sqrt(mean_squared_error(test_y, final_predictions))
        final_r2 = r2_score(test_y, final_predictions)
        final_mae = mean_absolute_error(test_y, final_predictions)

        print(f"Final Model RMSE: {final_rmse}, R2: {final_r2}, MAE: {final_mae}")

        mlflow.log_metric("final_rmse", final_rmse)
        mlflow.log_metric("final_r2_score", final_r2)
        mlflow.log_metric("final_mae", final_mae)

    # # Log the retraining history
    # if retrain_history is not None:
    #     for key, values in retrain_history.history.items():
    #         for i, val in enumerate(values):
    #             mlflow.log_metric(f"{key}_{i}", val)


