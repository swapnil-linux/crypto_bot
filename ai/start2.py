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


close_price = data.filter(['Close'])

dataset = close_price.values
training_data_len = int(np.ceil( len(dataset) * .8 ))

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data = scaler.fit_transform(data)
scaled_data = scaler.fit_transform(close_price)


train_data = scaled_data[0:int(training_data_len), :]

x_train=[]
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=20)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data (LSTM expects 3D data)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the model predicted price values
future_days = 60
pred_prices = []

# Use last 60 days from the test data for the first prediction
last_60_days = test_data[-60:]

for _ in range(future_days):
    # Reshape the data
    input_data = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    
    # Make a prediction
    predicted_price = model.predict(input_data)
    
    # Append the prediction to the pred_prices list
    pred_prices.append(predicted_price[0])
    
    # Append the prediction to input data for predicting next day
    last_60_days = np.append(last_60_days[1:], predicted_price)

# Convert the pred_prices list to an array
pred_prices = np.array(pred_prices)

# Create a dummy array with the same shape as the predictions
dummy = np.zeros((len(pred_prices), data.shape[1]))

# Place the predicted prices in the 'Close' column of the dummy array
dummy[:, 3] = np.squeeze(pred_prices)

# Inverse transform the dummy array
pred_prices = scaler.inverse_transform(dummy)[:, 3]

# Concatenate y_test and pred_prices to plot them together
none_array = np.array([None for _ in range(future_days)])
none_array = none_array.reshape(-1, 1)  # Reshape to 2D array
full_y_test = np.concatenate((y_test, none_array))
full_pred_prices = np.concatenate((np.full(y_test.shape, None), pred_prices.reshape(-1, 1)))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(full_y_test, color='blue', label=f'Actual {ticker} Stock Price')
plt.plot(full_pred_prices, color='red', label=f'Predicted {ticker} Stock Price')

plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()
plt.show()