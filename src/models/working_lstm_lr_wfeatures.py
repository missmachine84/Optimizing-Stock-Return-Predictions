import os
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.optimizers import Adam

# Function to split dataset into train and test sets
def split_dataset(data):
    train_size = int(len(data) * 0.8)  # 80% of data for training
    train, test = data[:train_size], data[train_size:]  # Split data into train and test sets
    return train, test


# Function to transform the dataset into supervised learning format
def to_supervised(train, n_input, n_out):
    X, y = list(), list()  # Initialize lists for inputs and outputs
    in_start = 0  # Starting index for input sequence
    for _ in range(len(train)):  # Loop over the dataset
        in_end = in_start + n_input  # Ending index for input sequence
        out_end = in_end + n_out  # Ending index for output sequence
        if out_end <= len(train):  # Ensure indices are within bounds
            X.append(train[in_start:in_end, :])  # Append input sequence
            y.append(train[in_end:out_end, -1])  # Append output sequence (last column)
        in_start += 1  # Move to the next sequence
    print(f"X shape: {np.array(X).shape}, y shape: {np.array(y).shape}")  # Print the shapes of X and y
    return np.array(X), np.array(y)  # Return input and output sequences as numpy arrays


# Function to build the LSTM model
def build_lstm_model(train, n_input, n_out):
    train_x, train_y = to_supervised(train, n_input, n_out)  # Prepare data for supervised learning
    verbose, epochs, batch_size = 0, 50, 16  # Set model parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]  # Get data dimensions
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))  # Reshape output data for LSTM

    model = Sequential()  # Initialize the Sequential model
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))  # Add LSTM layer
    model.add(RepeatVector(n_outputs))  # Repeat the vector n_outputs times
    model.add(LSTM(100, activation='relu', return_sequences=True))  # Add another LSTM layer
    model.add(TimeDistributed(Dense(100, activation='relu')))  # Add time-distributed Dense layer
    model.add(TimeDistributed(Dense(1)))  # Add time-distributed output layer
    optimizer = Adam(learning_rate=0.001)  # Set the optimizer
    model.compile(optimizer=optimizer, loss='mse')  # Compile the model with mean squared error loss
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)  # Train the model
    return model

# Function to build the Linear Regression model
def build_lr_model(train_x, train_y):
    train_x = train_x.reshape((train_x.shape[0], -1))  # Flatten input data for Linear Regression
    model = LinearRegression()  # Initialize the Linear Regression model
    model.fit(train_x, train_y)  # Train the model
    return model  # Return the trained model

# Function to forecast using LSTM model
def forecast_lstm(model, history, n_input):
    data = np.array(history)  # Convert history to numpy array
    input_x = data[-n_input:, :]  # Get the last n_input sequences
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))  # Reshape for LSTM input
    yhat = model.predict(input_x, verbose=0)  # Make prediction
    yhat = yhat[0]  # Get the predicted sequence
    return yhat

# Function to forecast using Linear Regression model
def forecast_lr(model, history, n_input):
    data = np.array(history)  # Convert history to numpy array
    input_x = data[-n_input:, :].reshape(1, -1)  # Get the last n_input sequences and flatten
    yhat = model.predict(input_x)  # Make prediction
    return yhat[0]  # Flatten to 1D array for correct shape

# Function to evaluate forecasts
def evaluate_forecasts(actual, predicted):
    scores = list()  # Initialize list to store RMSE for each forecast
    for i in range(actual.shape[1]):  # Loop over each forecast
        mse = mean_squared_error(actual[:, i], predicted[:, i])  # Calculate mean squared error
        rmse = sqrt(mse)  # Calculate root mean squared error
        scores.append(rmse)  # Append RMSE to scores
    s = 0  # Initialize sum for overall RMSE
    for row in range(actual.shape[0]):  # Loop over each actual value
        for col in range(actual.shape[1]):  # Loop over each forecast value
            s += (actual[row, col] - predicted[row, col]) ** 2  # Sum squared errors
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))  # Calculate overall RMSE
    return score, scores  # Return overall RMSE and list of RMSE for each forecast

# Function to evaluate model performance
def evaluate_model(train, test, n_input, n_out, model_type='lstm'):
    if model_type == 'lstm':  # Check if model type is LSTM
        model = build_lstm_model(train, n_input, n_out)  # Build LSTM model
        forecast = forecast_lstm  # Set forecast function to LSTM forecast
    elif model_type == 'lr':  # Check if model type is Linear Regression
        train_x, train_y = to_supervised(train, n_input, n_out)  # Prepare data for supervised learning
        model = build_lr_model(train_x.reshape((train_x.shape[0], -1)), train_y.reshape((train_y.shape[0], -1)))  # Build Linear Regression model
        forecast = forecast_lr  # Set forecast function to Linear Regression forecast

    history = [x for x in train]  # Initialize history with training data
    predictions = list()  # Initialize list to store predictions
    for i in range(len(test) - n_out + 1):  # Loop over test data
        yhat_sequence = forecast(model, history, n_input)  # Make forecast
        predictions.append(yhat_sequence)  # Append forecast to predictions
        history.append(test[i, :])  # Update history with current test data

    predictions = np.array(predictions)  # Convert predictions to numpy array
    actual = np.array([test[i:i + n_out, -1] for i in range(len(test) - n_out + 1)])  # Get actual values
    score, scores = evaluate_forecasts(actual, predictions)  # Evaluate forecasts
    return model, score, scores, predictions  # Return model, overall RMSE, list of RMSE, and predictions

# Function to plot predictions
def plot_predictions(stock_name, weeks, train, test, lstm_predictions, lr_predictions, train_smape, train_rmse, test_smape, test_rmse):
    actual = np.concatenate((train[:, -1], test[:, -1]))  # Concatenate train and test actual values

    # Flatten LSTM predictions and add NaNs to align with the actual values
    lstm_predictions_flat = lstm_predictions.flatten()  # Flatten LSTM predictions
    lstm_predicted = np.concatenate((np.full(len(train) - n_input, np.nan), lstm_predictions_flat))  # Add NaNs to align

    # Ensure the lengths match
    if len(lstm_predicted) < len(actual):  # Check if LSTM predictions are shorter than actual values
        lstm_predicted = np.concatenate((lstm_predicted, np.full(len(actual) - len(lstm_predicted), np.nan)))  # Add NaNs
    elif len(lstm_predicted) > len(actual):  # Check if LSTM predictions are longer than actual values
        lstm_predicted = lstm_predicted[:len(actual)]  # Trim predictions

    # Flatten LR predictions and add NaNs to align with the actual values
    lr_predictions_flat = lr_predictions.flatten()  # Flatten LR predictions
    lr_predicted = np.concatenate((np.full(len(train) - n_input, np.nan), lr_predictions_flat))  # Add NaNs to align

    # Ensure the lengths match
    if len(lr_predicted) < len(actual):  # Check if LR predictions are shorter than actual values
        lr_predicted = np.concatenate((lr_predicted, np.full(len(actual) - len(lr_predicted), np.nan)))  # Add NaNs
    elif len(lr_predicted) > len(actual):  # Check if LR predictions are longer than actual values
        lr_predicted = lr_predicted[:len(actual)]  # Trim predictions

    # Print lengths and indices for debugging
    print(f'Length of actual values: {len(actual)}')
    print(f'Length of LSTM predicted values: {len(lstm_predicted)}')
    print(f'Length of LR predicted values: {len(lr_predicted)}')

    # Print the date when prediction starts
    prediction_start_date = weeks[len(train) + n_input]  # Get the start date of predictions
    print(f'{stock_name} LSTM prediction starts at: {prediction_start_date}')
    print(f'{stock_name} LR prediction starts at: {prediction_start_date}')

    # Update the plot title with SMAPE and RMSE for both training and test sets
    title_with_scores = (
        f"LSTM and LR: {stock_name} Actual vs Predicted Returns\n"
        f"Training SMAPE: {train_smape:.2f}%, Training RMSE: {train_rmse:.2f}\n"
        f"Prediction SMAPE: {test_smape:.2f}%, Prediction RMSE: {test_rmse:.2f}"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(weeks, actual, label='Actual')  # Plot actual values
    plt.plot(weeks, lstm_predicted, label='LSTM Predicted', linestyle='--')  # Plot LSTM predictions
    plt.plot(weeks, lr_predicted, label='LR Predicted', linestyle='-.')  # Plot LR predictions
    plt.title(title_with_scores)
    plt.xlabel('Week')
    plt.ylabel('Value')
    plt.legend()

    # Spread ticks out
    max_ticks = 10  # Maximum number of ticks to display
    tick_frequency = max(1, len(weeks) // max_ticks)  # Calculate tick frequency
    plt.xticks(ticks=range(0, len(weeks), tick_frequency), labels=weeks[::tick_frequency], rotation=45)  # Set x-ticks

    plt.tight_layout()  # Adjust layout to make room for tick labels
    plt.savefig(os.path.join(output_folder, f'LSTM_LR_{stock_name}_prediction.png'))
    plt.close()

# Function to calculate SMAPE
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))  # Calculate SMAPE

# Main script: Load and process data
input_folder = 'stock_data'
output_folder = 'Model_Visualizations'
n_input = 4  # Number of input sequences
n_out = 1  # Number of output sequences

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

all_datasets = {}  # Initialize dictionary to store datasets
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(input_folder, filename)
        if os.path.isfile(filepath):
            stock_name = os.path.splitext(filename)[0]
            dataset = pd.read_csv(filepath)
            weeks = dataset['Week'].values
            dataset = dataset.drop(columns=['Week'])
            dataset = dataset.apply(pd.to_numeric, errors='coerce').dropna().values
            train, test = split_dataset(dataset)
            all_datasets[stock_name] = (weeks, train, test)
        else:
            print(f"File not found: {filepath}")

# Initialize variables to calculate overall SMAPE and RMSE scores
overall_train_smape = []
overall_train_rmse = []
overall_test_smape = []
overall_test_rmse = []
combined_smape = []
combined_rmse = []

# Process each dataset
for stock_name, (weeks, train, test) in all_datasets.items():
    print(f'Processing stock: {stock_name}')

    lstm_model, lstm_score, lstm_scores, lstm_predictions = evaluate_model(train, test, n_input, n_out, model_type='lstm')  # Evaluate LSTM model
    print(f'LSTM Model RMSE: {lstm_score}')

    lr_model, lr_score, lr_scores, lr_predictions = evaluate_model(train, test, n_input, n_out, model_type='lr')  # Evaluate Linear Regression model
    print(f'LR Model RMSE: {lr_score}')

    train_x, train_y = to_supervised(train, n_input, n_out)  # Prepare data for supervised learning
    train_predictions = lstm_model.predict(train_x).reshape(-1)  # Make LSTM predictions on training data
    train_y_flat = train_y.reshape(-1)[:len(train_predictions)]  # Flatten training outputs
    train_smape = smape(train_y_flat, train_predictions)  # Calculate SMAPE on training data
    train_rmse = sqrt(mean_squared_error(train_y_flat, train_predictions))  # Calculate RMSE on training data

    test_actual = np.array([test[i:i + n_out, -1] for i in range(len(test) - n_out + 1)]).reshape(-1)  # Get actual test values
    test_smape = smape(test_actual, lstm_predictions.reshape(-1))  # Calculate SMAPE on test data
    test_rmse = sqrt(mean_squared_error(test_actual, lstm_predictions.reshape(-1)))  # Calculate RMSE on test data

    combined_actual = np.concatenate((train_y_flat, test_actual))  # Combine training and test actual values
    combined_predictions = np.concatenate((train_predictions, lstm_predictions.reshape(-1)))  # Combine training and test predictions

    combined_score_smape = smape(combined_actual, combined_predictions)  # Calculate combined SMAPE
    combined_score_rmse = sqrt(mean_squared_error(combined_actual, combined_predictions))  # Calculate combined RMSE

    print(f'Training SMAPE: {train_smape:.2f}%')
    print(f'Training RMSE: {train_rmse:.2f}')
    print(f'Prediction SMAPE: {test_smape:.2f}%')
    print(f'Prediction RMSE: {test_rmse:.2f}')
    print(f'Combined SMAPE: {combined_score_smape:.2f}%')
    print(f'Combined RMSE: {combined_score_rmse:.2f}')

    # Add the scores to the overall lists
    overall_train_smape.append(train_smape)
    overall_train_rmse.append(train_rmse)
    overall_test_smape.append(test_smape)
    overall_test_rmse.append(test_rmse)
    combined_smape.append(combined_score_smape)
    combined_rmse.append(combined_score_rmse)

    plot_predictions(stock_name, weeks, train, test, lstm_predictions, lr_predictions, train_smape, train_rmse, test_smape, test_rmse)  # Plot predictions

# Calculate overall SMAPE and RMSE scores
avg_train_smape = np.mean(overall_train_smape)
avg_train_rmse = np.mean(overall_train_rmse)
avg_test_smape = np.mean(overall_test_smape)
avg_test_rmse = np.mean(overall_test_rmse)
avg_combined_smape = np.mean(combined_smape)
avg_combined_rmse = np.mean(combined_rmse)

print(f'Overall Training SMAPE: {avg_train_smape:.2f}%')
print(f'Overall Training RMSE: {avg_train_rmse:.2f}')
print(f'Overall Prediction SMAPE: {avg_test_smape:.2f}%')
print(f'Overall Prediction RMSE: {avg_test_rmse:.2f}')
print(f'Overall Combined SMAPE: {avg_combined_smape:.2f}%')
print(f'Overall Combined RMSE: {avg_combined_rmse:.2f}')
