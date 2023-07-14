# pip install yfinance pycoingecko prettytable ta tensorflow pandas scikit-learn

import yfinance as yf
from pycoingecko import CoinGeckoAPI
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd

cg = CoinGeckoAPI()

def get_top_crypto_symbols(n):
    top_20_cryptos = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=n)
    symbols = [crypto['symbol'].upper() + '-USD' for crypto in top_20_cryptos]
    return symbols


# Define the scalers outside the function
scaler_close = StandardScaler()
scaler_volume = StandardScaler()

def prepare_data(data, look_forward=5):
    # Scaling the input data
    global scaler_close
    global scaler_volume
    close_scaled = scaler_close.fit_transform(data[['Close']])
    volume_scaled = scaler_volume.fit_transform(data[['Volume']])
    data_scaled = np.hstack((close_scaled, volume_scaled))

    # Create features and labels
    X = []
    y = []
    for i in range(60, len(data_scaled) - look_forward):
        X.append(data_scaled[i-60:i])
        # Calculate the price change
        price_now = data_scaled[i, 0]
        price_future = data_scaled[i + look_forward, 0]
        # Set label to 1 if price goes up in the next 'look_forward' days, else 0
        y.append(1 if price_future > price_now else 0)

    return np.array(X), np.array(y)



def build_model(input_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(128, input_shape=input_shape, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    # Add a flatten layer before the final dense layer
    model.add(tf.keras.layers.Flatten())

    # The final layer should output a single value
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



# Define list of tickers
tickers = ['AAPL','TSLA']

# Initialize the table
table = PrettyTable()
table.field_names = ["Ticker", "Last price", "Signal"]

table_err = PrettyTable()
table_err.field_names = ["Ticker", "Error"]

# Loop over tickers
with tqdm(total=len(tickers), unit="ticker") as pbar:
    for i, ticker in enumerate(tickers):
        pbar.set_description(f"Downloading data for {ticker}")

        try:
            data = yf.download(ticker, period="10y", interval="1d", progress=False)
            
            # Preparing data
            X, y = prepare_data(data)

            # Scaling the input data
            scaler = StandardScaler()
            X_scaled = np.array([scaler.fit_transform(x) for x in X])

            # Splitting into train and test
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Building the model
            model = build_model(X_train.shape[1:])
            
            # Training the model with dummy labels
            model.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(X_test, y_test))

            # Using the model to predict
            last_data = data[['Close', 'Volume']].values[-60:]
            last_close_scaled = scaler_close.transform(pd.DataFrame(last_data[:,0], columns=["Close"]))
            last_volume_scaled = scaler_volume.transform(pd.DataFrame(last_data[:,1], columns=["Volume"]))
            last_data_scaled = np.hstack((last_close_scaled, last_volume_scaled)).reshape(1, 60, 2)
            prediction = model.predict(last_data_scaled)




            # Adding result to the table
            last_price = round(data['Close'][-1], 2)
            signal = "Buy" if prediction > 0.5 else "Sell"
            table.add_row([ticker, last_price, signal])

            
            pbar.update(1)
        except Exception as e:
            table_err.add_row([ticker, str(e)])

print(table)
print(table_err)
