import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to calculate additional features
def calculate_features(data):
    data['prev_day_diff'] = data['Adj Close'].diff()
    data['50_day_moving_avg'] = data['Adj Close'].rolling(window=50, min_periods=1).mean()
    data['10_day_volatility'] = data['Adj Close'].rolling(window=10, min_periods=1).std()
    return data

# Load the input CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.sort_values('Date', inplace=True)
    data = calculate_features(data)
    data.dropna(inplace=True)
    return data

# Prepare features and targets
def prepare_data(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'prev_day_diff', '50_day_moving_avg', '10_day_volatility']]
    target_high = data['High'].shift(-1).fillna(data['High'])
    target_low = data['Low'].shift(-1).fillna(data['Low'])
    return features, target_high, target_low

# Train the model and make predictions
def train_and_predict(features, target_high, target_low):
    X_train, X_test, y_train_high, y_test_high = train_test_split(features, target_high, test_size=0.2, random_state=42)
    _, _, y_train_low, y_test_low = train_test_split(features, target_low, test_size=0.2, random_state=42)

    model_high = RandomForestRegressor(random_state=42)
    model_low = RandomForestRegressor(random_state=42)

    model_high.fit(X_train, y_train_high)
    model_low.fit(X_train, y_train_low)

    predictions_high = model_high.predict(X_test)
    predictions_low = model_low.predict(X_test)

    return y_test_high, predictions_high, y_test_low, predictions_low

# Calculate accuracy
def calculate_accuracy(y_test_high, predictions_high, y_test_low, predictions_low):
    high_mape = mean_absolute_percentage_error(y_test_high, predictions_high) * 100
    low_mape = mean_absolute_percentage_error(y_test_low, predictions_low) * 100

    high_accuracy = 100 - high_mape
    low_accuracy = 100 - low_mape

    return high_accuracy, low_accuracy

# Plot the predictions and actual values
def plot_results(y_test_high, predictions_high, y_test_low, predictions_low):
    plt.figure(figsize=(12, 6))
    
    # High prices
    plt.subplot(1, 2, 1)
    plt.plot(y_test_high.values, label='Actual High', marker='o')
    plt.plot(predictions_high, label='Predicted High', linestyle='--', marker='x')
    plt.title('Actual vs Predicted High Prices')
    plt.xlabel('Test Data Points')
    plt.ylabel('Price')
    plt.legend()
    
    # Low prices
    plt.subplot(1, 2, 2)
    plt.plot(y_test_low.values, label='Actual Low', marker='o')
    plt.plot(predictions_low, label='Predicted Low', linestyle='--', marker='x')
    plt.title('Actual vs Predicted Low Prices')
    plt.xlabel('Test Data Points')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function
def main(file_path):
    data = load_data(file_path)
    features, target_high, target_low = prepare_data(data)
    y_test_high, predictions_high, y_test_low, predictions_low = train_and_predict(features, target_high, target_low)
    high_accuracy, low_accuracy = calculate_accuracy(y_test_high, predictions_high, y_test_low, predictions_low)

    # Output results
    output_data = pd.DataFrame({
        'Symbol': data['Symbol'].iloc[:len(predictions_high)],
        'Date': data['Date'].iloc[:len(predictions_high)],
        'Prev Day Diff': data['prev_day_diff'].iloc[:len(predictions_high)],
        '50 Day Moving Avg': data['50_day_moving_avg'].iloc[:len(predictions_high)],
        '10 Day Volatility': data['10_day_volatility'].iloc[:len(predictions_high)],
        'Predicted Next Day High': predictions_high,
        'Predicted Next Day Low': predictions_low
    })

    print(output_data)
    print(f"High Prediction Accuracy: {high_accuracy:.2f}%")
    print(f"Low Prediction Accuracy: {low_accuracy:.2f}%")

    # Plot predictions
    plot_results(y_test_high, predictions_high, y_test_low, predictions_low)

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'bmy.csv'
main(file_path)
