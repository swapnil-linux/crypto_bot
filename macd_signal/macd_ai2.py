import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from prettytable import PrettyTable
from ta.trend import MACD
from tqdm import tqdm

# Initialize coingecko API
cg = CoinGeckoAPI()

def get_top_crypto_symbols(n):
    top_20_cryptos = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=n)
    symbols = [crypto['symbol'].upper() + '-USD' for crypto in top_20_cryptos]
    return symbols

# Define list of tickers
tickers = ['AAPL', 'ABBV', 'ABNB', 'ABT', 'ACN', 'ADANIPORTS.NS', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALGN', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMZN', 'ANSS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'ASML', 'ATVI', 'AVGO', 'AXISBANK.NS', 'AXP', 'AZN', 'BA', 'BAC', 'BAJAJFINSV.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BIIB', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BPCL.NS', 'BRITANNIA.NS', 'BRK-B', 'C', 'CAT', 'CDNS', 'CEG', 'CHTR', 'CIPLA.NS', 'CL', 'CMCSA', 'COALINDIA.NS', 'COF', 'COP', 'COST', 'CPRT', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CVS', 'CVX', 'DDOG', 'DHR', 'DIS', 'DIVISLAB.NS', 'DLTR', 'DOW', 'DRREDDY.NS', 'DUK', 'DXCM', 'EA', 'EBAY', 'EICHERMOT.NS', 'EMR', 'ENPH', 'EXC', 'F', 'FANG', 'FAST', 'FDX', 'FISV', 'FTNT', 'GD', 'GE', 'GFS', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GRASIM.NS', 'GS', 'HCLTECH.NS', 'HD', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'HON', 'IBM', 'ICICIBANK.NS', 'IDXX', 'ILMN', 'INDUSINDBK.NS', 'INFY.NS', 'INTC', 'INTU', 'ISRG', 'ITC.NS', 'JD', 'JNJ', 'JPM', 'JSWSTEEL.NS', 'KDP', 'KHC', 'KLAC', 'KO', 'KOTAKBANK.NS', 'LCID', 'LIN', 'LLY', 'LMT', 'LOW', 'LRCX', 'LT.NS', 'LULU', 'MA', 'MAR', 'MARUTI.NS', 'MCD', 'MCHP', 'MDLZ', 'MDT', 'MELI', 'META', 'MMM', 'MNST', 'MO', 'MRK', 'MRNA', 'MRVL', 'MS', 'MSFT', 'MU', 'M&M.NS', 'NEE', 'NESTLEIND.NS', 'NFLX', 'NKE', 'NTPC.NS', 'NVDA', 'NXPI', 'ODFL', 'ONGC.NS', 'ORCL', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PFE', 'PG', 'PM', 'POWERGRID.NS', 'PYPL', 'QCOM', 'REGN', 'RELIANCE.NS', 'RIVN', 'ROST', 'RTX', 'SBILIFE.NS', 'SBIN.NS', 'SBUX', 'SCHW', 'SGEN', 'SIRI', 'SNPS', 'SO', 'SPG', 'SUNPHARMA.NS', 'T', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TEAM', 'TECHM.NS', 'TGT', 'TITAN.NS', 'TMO', 'TMUS', 'TSLA', 'TXN', 'ULTRACEMCO.NS', 'UNH', 'UNP', 'UPL.NS', 'UPS', 'USB', 'V', 'VRSK', 'VRTX', 'VZ', 'WBA', 'WBD', 'WDAY', 'WFC', 'WIPRO.NS', 'WMT', 'XEL', 'XOM', 'ZM', 'ZS']
#tickers = ['AAPL']
#cryptos = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'MATIC-USD', 'DOT-USD', 'SOL-USD', 'LTC-USD', 'SHIB-USD', 'CRO-USD', 'VET-USD', 'FTM-USD', 'BAKE-USD', 'WAVES-USD', 'AGLD-USD', 'CHR-USD' ]
#cryptos = ['BTC-USD']
# Add top 20 crypto symbols to the stock_tickers list
cryptos = get_top_crypto_symbols(20)
tickers.extend(cryptos)

# Define your model
def build_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess data and create datasets
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)  
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Initialize the table
table = PrettyTable()
table.field_names = ["Ticker", "Last price", "Signal"]

table_sell = PrettyTable()
table_sell.field_names = ["Ticker", "Last price", "Signal"]

table_err = PrettyTable()
table_err.field_names = ["Ticker", "Error"]

with tqdm(total=len(tickers), unit="ticker") as pbar:
    for i, ticker in enumerate(tickers):
        pbar.set_description(f"Downloading data for {ticker}")

        try:
            # Download historical data as dataframe
            data = yf.download(ticker, period="1y", interval="1d", progress=False)

            # Calculate MACD
            macd = MACD(data['Close'])
            signal_line = macd.macd_signal()
            macd_line = macd.macd()

            # Create labels based on MACD crossovers
            data['Label'] = np.where((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 1, 0)

            # Normalize the data
            features = data['Close'].values.reshape(-1,1)
            labels = data['Label'].values

            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_features = scaler.fit_transform(features)

            # Create the datasets
            time_steps = 60
            X, y = create_dataset(pd.DataFrame(scaled_features), labels, time_steps)


            # Reshape the labels to match the output shape of the model
            y = y.reshape(-1, 1)

            # Initialize and train the model
            model = build_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, shuffle=False, verbose=0)

            # Get the most recent data
            recent_data = scaled_features[-time_steps:]
            recent_data = np.array(recent_data)
            recent_data = np.reshape(recent_data, (1, recent_data.shape[0], 1))


            # Predict on the most recent data
            pred = model.predict(recent_data)

            # Assuming pred is a 2D array but with a single row of predictions
            pred_value = pred[0][0]
            last_price = round(data['Close'][-1], 2)

            if pred_value > 0.5:
                table.add_row([ticker, last_price, "Buy"])
            else:
                table_sell.add_row([ticker, last_price, "Sell"])
            pbar.update(1)   
        except KeyError:
            table_err.add_row([ticker, "Data not found"])
        except IndexError:
            table_err.add_row([ticker, "Empty DataFrame"])
        except Exception as e:
            table_err.add_row([ticker, f"Error: {str(e)}"])

print(table)
print(table_sell)
