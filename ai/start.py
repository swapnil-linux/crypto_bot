import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
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

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Compute the number of rows to train the model on
training_data_len = int(np.ceil(len(scaled_data) * .8))

# Split the data into training and test sets
train_data = scaled_data[0:training_data_len, :]
test_data = scaled_data[training_data_len - 60:, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, :])
    y_train.append(train_data[i, 3])  # 'Close' price is at index 3

x_train, y_train = np.array(x_train), np.array(y_train)

# Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=50)

# Create the data sets x_test and y_test
x_test = []
y_test = data['Close'][training_data_len:].values
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Get the model predicted price values
pred_price = model.predict(x_test)

# Create a dummy array with the same shape as 'test_data'
dummy = np.zeros(shape=(len(pred_price), data.shape[1]))

# Place the predicted prices in the 'Close' column of the dummy array
dummy[:,3] = np.squeeze(pred_price)

# Inverse transform the dummy array
pred_price = scaler.inverse_transform(dummy)[:,3]

# Create the input_data
input_data = scaled_data[-60:].copy()

# Empty list to store the predictions
predictions = []

# Append the last 'n' days data to the input
for i in range(20):
    # Reshape and append the input data for the model
    input_data = input_data[-60:]
    input_data_reshaped = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))

    # Predict the price
    pred_price = model.predict(input_data_reshaped)

    # Create a dummy array with the same shape as 'input_data'
    dummy = np.zeros(shape=(1, data.shape[1]))

    # Place the predicted prices in the 'Close' column of the dummy array
    dummy[:,3] = np.squeeze(pred_price)

    # Append the predicted price to the list and input data
    predictions.append(np.squeeze(pred_price))  # Squeeze the prediction to get the scalar value
    input_data = np.concatenate((input_data, dummy), axis=0)

# Create a dummy array with the same shape as 'predictions'
dummy = np.zeros(shape=(len(predictions), data.shape[1]))

# Place the predicted prices in the 'Close' column of the dummy array
dummy[:,3] = np.squeeze(predictions)

# Inverse scale the predictions
predictions = scaler.inverse_transform(dummy)[:,3]

# Actual prices for the last 'n' days

past_days = 20
future_days = 20


#actual_prices = data['Close'][-20:].values
actual_prices = data['Close'][-past_days:].resample('B').asfreq().dropna().values


# Create a timeline for the actual prices
timeline_actual = pd.date_range(start=data.index[-past_days], periods=len(actual_prices), freq='B')

# Create a timeline for the predicted prices. Start from the next business day after the last day of actual data
timeline_predicted = pd.date_range(start=data.index[-1] + BDay(1), periods=future_days, freq='B')

plt.figure(figsize=(10,6))

# Plot actual prices
plt.plot(timeline_actual, actual_prices, color='blue', label=f'Actual {ticker} Stock Price')

# Plot predicted prices
plt.plot(timeline_predicted, predictions[-future_days:], color='red', label=f'Predicted {ticker} Stock Price')

plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()
plt.show()
