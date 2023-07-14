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

cg = CoinGeckoAPI()

def get_top_crypto_symbols(n):
    top_20_cryptos = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=n)
    symbols = [crypto['symbol'].upper() + '-USD' for crypto in top_20_cryptos]
    return symbols


# Define the scalers outside the function
scaler_close = StandardScaler()
scaler_volume = StandardScaler()

def prepare_data(data, look_forward=5):
    global scaler_close
    global scaler_volume
    close_scaled = scaler_close.fit_transform(data[['Close']])
    volume_scaled = scaler_volume.fit_transform(data[['Volume']])
    data_scaled = np.hstack((close_scaled, volume_scaled))

    X = []
    y = []
    for i in range(60, len(data_scaled) - look_forward):
        X.append(data_scaled[i-60:i])
        price_now = data_scaled[i, 0]
        price_future = data_scaled[i + look_forward, 0]
        y.append(1 if price_future > price_now else 0)

    return np.array(X), np.array(y)


def build_model(input_shape):
    model = tf.keras.models.Sequential()

    # Increase model complexity by adding more layers
    model.add(tf.keras.layers.Dense(256, input_shape=input_shape, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    
    # Add a flatten layer before the final dense layer
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Use an advanced optimizer
    model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

    return model


tickers = ['AAPL','TSLA']

table = PrettyTable()
table.field_names = ["Ticker", "Last price", "Signal"]

table_err = PrettyTable()
table_err.field_names = ["Ticker", "Error"]

with tqdm(total=len(tickers), unit="ticker") as pbar:
    for i, ticker in enumerate(tickers):
        pbar.set_description(f"Downloading data for {ticker}")

        try:
            data = yf.download(ticker, period="10y", interval="1d", progress=False)
            
            X, y = prepare_data(data)
            scaler = StandardScaler()
            X_scaled = np.array([scaler.fit_transform(x) for x in X])

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = build_model(X_train.shape[1:])

            # Define early stopping criteria
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

            model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1, 
                      validation_data=(X_test, y_test), callbacks=[early_stopping])

            last_data = data[['Close', 'Volume']].values[-60:].reshape(1, 60, 2)
            last_data_scaled = np.hstack((scaler_close.transform(last_data[:,:,0].reshape(-1,1)), scaler_volume.transform(last_data[:,:,1].reshape(-1,1))))
            prediction = model.predict(last_data_scaled)

            last_price = round(data['Close'][-1], 2)
            signal = "Buy" if prediction > 0.5 else "Sell"
            table.add_row([ticker, last_price, signal])

            pbar.update(1)
        except Exception as e:
            table_err.add_row([ticker, str(e)])

print(table)
print(table_err)
