import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras import regularizers
import matplotlib.pyplot as plt
import yfinance as yf
import ta

# ticker = input("Enter ticker symbol: ")
ticker = "UI"
data = yf.download(ticker, period="5y", interval="1d", progress=True)

print("Calculate Simple Moving Average (SMA)")
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)

print("Calculate Exponential Moving Average (EMA)")
data['EMA'] = ta.trend.ema_indicator(data['Close'], window=14)

print("Calculate Relative Strength Index (RSI)")
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

print("Calculate MACD")
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd_diff()

# Calculate Average True Range (ATR)
print("Calculate Average True Range (ATR)")
data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

# Calculate Stochastic Oscillator
print("Calculate Stochastic Oscillator")
stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
data['%K'] = stoch.stoch()
data['%D'] = stoch.stoch_signal()

# Calculate Bollinger Bands
print("Calculate Bollinger Bands")
bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
data['Bollinger_hband'] = bollinger.bollinger_hband()
data['Bollinger_lband'] = bollinger.bollinger_lband()

print("Download S&P 500 data")
sp500_data = yf.download('^GSPC', period="5y", interval="1d", progress=True)

# Only keep the 'Close' column of the S&P 500 data
sp500_data = sp500_data[['Close']].rename(columns={'Close': 'SP500'})

print("Merge the dataframes on the date index")
data = pd.concat([data, sp500_data], axis=1)

# Drop the first 33 rows which have NaN values due to MACD calculation
data = data.dropna()
# print(data)

print("Scale the data")

# Define the length of the training data
training_data_len = int(np.ceil(len(data) * 0.8))

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))

# Fit the scaler only on the training data portion
scaler.fit(data.iloc[:training_data_len])

# Transform the entire dataset
scaled_data = scaler.transform(data)

# Extract the Close prices and scale them separately
close_scaler = MinMaxScaler(feature_range=(0,1))
close_price = data.filter(['Close'])
scaled_close = close_scaler.fit_transform(close_price.iloc[:training_data_len])

print("Prepare training dataset")
train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i])
    y_train.append(scaled_close[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print("Create LSTM model")

model = Sequential()

# First LSTM layer with L2 regularization
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

# Third LSTM layer
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Dense layers
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=25, activation='relu'))

# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=50)

print("Create the testing data set")
test_data = scaled_data[training_data_len - 60:, :]
print(test_data)
# Create the data sets x_test and y_test
x_test = []
y_test = close_price.values[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i])

# Convert the data to a numpy array
x_test = np.array(x_test)

print("Get the model predicted price values")
pred_prices = model.predict(x_test)

# Inverse transform the predicted prices
pred_prices = close_scaler.inverse_transform(pred_prices)

# Future predictions
future_days = 10
future_pred_prices = []

# Input for the first prediction is the last 60 days of known data
input_data = scaled_data[-60:]

# Iterate over the range of future days to be predicted
for _ in range(future_days):
    # Reshape the input_data for making a prediction
    reshaped_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))

    # Predict the next price
    future_predicted_price = model.predict(reshaped_data)
    
    # Inverse transform the predicted price
    future_predicted_price_unscaled = close_scaler.inverse_transform(future_predicted_price)[0, 0]
    
    # Append future_predicted_price to future_pred_prices
    future_pred_prices.append(future_predicted_price_unscaled)
    
    # Prepare new_input_data
    new_input_data = np.zeros((1, input_data.shape[1]))
    
    # Populate the 'Close' feature in new_input_data with the predicted price
    new_input_data[0, 3] = future_predicted_price[0, 0]
    
    # For simplicity, populate the other features with the values from the last day
    # This is a naive approach and you might want to use a more sophisticated strategy for the other features
    new_input_data[0, :3] = input_data[-1, :3]
    new_input_data[0, 4:] = input_data[-1, 4:]
    
    # Stack them together
    input_data = np.vstack((input_data[1:], new_input_data))


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label=f'Actual {ticker} Stock Price')
plt.plot(np.arange(len(pred_prices)), pred_prices, color='red', label=f'Predicted {ticker} Stock Price')
plt.plot(np.arange(len(pred_prices), len(pred_prices) + future_days), future_pred_prices, color='green', label=f'Future {ticker} Stock Price')

plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()
# plt.show()
print(future_pred_prices)