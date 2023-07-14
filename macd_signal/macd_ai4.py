import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define MACD calculation
def calculate_macd(data, short_term=12, long_term=26):
    data["Short_EMA"] = data["Close"].ewm(span=short_term, adjust=False).mean()
    data["Long_EMA"] = data["Close"].ewm(span=long_term, adjust=False).mean()
    data["MACD"] = data["Short_EMA"] - data["Long_EMA"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data

# Define prediction model
def train_model(data):
    data["MACD_Above_Signal"] = np.where(data["MACD"] > data["Signal_Line"], 1, 0)
    data["MACD_Above_Signal_Previous"] = data["MACD_Above_Signal"].shift(1)
    data["Price_Increase_Next_Day"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    
    X = data.dropna()[["MACD_Above_Signal_Previous"]]
    y = data.dropna()["Price_Increase_Next_Day"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Get S&P100 symbols
sp100 = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'ALL', 'AMGN', 'AMT', 'AMZN',
         'AXP', 'BA', 'BAC', 'BIIB', 'BK', 'BKNG', 'BLK', 'BMY', 'C', 'CAT',
         'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX',
         'DD', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'GD', 'GE',
         'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM',
         'KHC', 'KMI', 'KO', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET',
         'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'OXY',
         'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SLB', 'SO', 'SPG',
         'T', 'TGT', 'TMO', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC',
         'WMT', 'XOM']

# Create an empty list to store stocks that meet condition
stocks_meeting_condition = []

for symbol in sp100:
    print(f"Processing {symbol}...")
    # Download historical data
    data = yf.download(symbol, period="1y", interval="1d", progress=False)

    # Calculate MACD
    data = calculate_macd(data)

    # Train and predict
    model = train_model(data)

    data["Prediction"] = model.predict(data[["MACD_Above_Signal_Previous"]].fillna(0))

    # Check condition
    data["Condition_Met"] = ((data["MACD"] < 0) &
                             (data["MACD"] > data["Signal_Line"]) &
                             (data["MACD"].shift(1) < data["Signal_Line"].shift(1)) &
                             (data["Prediction"] == 1))

    if data["Condition_Met"].any():
        stocks_meeting_condition.append(symbol)

print("Stocks meeting condition:")
print(stocks_meeting_condition)
