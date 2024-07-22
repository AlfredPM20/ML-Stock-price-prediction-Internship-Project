import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def compute_rsi(data, period=14):
    """Compute the Relative Strength Index (RSI) for a given data series."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fetch historical stock data for a specific stock
ticker = 'AAPL'  # Example: Apple Inc.
start_date = '2023-01-01'
end_date = '2028-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

# Feature Engineering
data['Moving_Avg_10'] = data['Close'].rolling(window=10).mean()
data['Moving_Avg_50'] = data['Close'].rolling(window=50).mean()
data['RSI'] = compute_rsi(data['Close'])

# Drop rows with missing values
data.dropna(inplace=True)

# Define features and target
features = data[['Moving_Avg_10', 'Moving_Avg_50', 'RSI']]
target = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(data.index[-len(y_test):], predictions, label='Predicted Prices', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
