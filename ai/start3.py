import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yf
import ta

ticker = input("Enter ticker symbol: ")

data = yf.download(ticker, period="3y", interval="1d", progress=True)

# Calculate Simple Moving Average (SMA)
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)

# Calculate Exponential Moving Average (EMA)
data['EMA'] = ta.trend.ema_indicator(data['Close'], window=14)

# Calculate Relative Strength Index (RSI)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# Calculate MACD
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd_diff()

# Drop the first 33 rows which have NaN values due to MACD calculation
data = data.dropna()

# Extract the close_price
close_price = data.filter(['Close'])

# Scale the entire data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Scale the close_price for plotting purposes
close_scaler = MinMaxScaler(feature_range=(0,1))
scaled_close = close_scaler.fit_transform(close_price)

# Split the data into training and testing sets
training_data_len = int(np.ceil(len(scaled_data) * .8))
train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=20)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = close_price.values[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Get the model predicted price values
pred_prices = model.predict(x_test)

# Inverse transform the predicted prices
pred_prices = close_scaler.inverse_transform(pred_prices)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label=f'Actual {ticker} Stock Price')
plt.plot(pred_prices, color='red', label=f'Predicted {ticker} Stock Price')

plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()
plt.show()

