import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def download_data(ticker='AUDUSD=X', interval='5m', period='1mo'):
    df = yf.download(ticker, interval=interval, period=period)
    df['Volume'] = df['Volume'].replace(to_replace=0, method='ffill')
    return df

def calculate_additional_indicators(df):
    df['SMA'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch'] = stoch.stoch()
    df['Stoch_signal'] = stoch.stoch_signal()
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    return df

def calculate_indicators(df):
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    bb = BollingerBands(df['Close'])
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['BB_mavg'] = bb.bollinger_mavg()
    df = calculate_additional_indicators(df)
    return df

def create_labels(df, take_profit=0.0005):
    df['Future_Close'] = df['Close'].shift(-1)
    df['Price_Change'] = df['Future_Close'] - df['Close']
    df['Target'] = np.where(df['Price_Change'] > take_profit, 1, 0)
    return df

def prepare_data(df):
    df = df.dropna()
    features = df[['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_high', 'BB_low', 'BB_mavg', 'SMA', 'EMA', 'ATR', 'Stoch', 'Stoch_signal', 'OBV']]
    target = df['Target']
    return features, target, df.index

def resample_data(features, target):
    smote = SMOTE()
    features_resampled, target_resampled = smote.fit_resample(features, target)
    return features_resampled, target_resampled

def train_model(features, target):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    return model

def find_entry_points(df, model):
    features, _, indices = prepare_data(df)
    predictions = model.predict(features)
    df.loc[indices, 'Prediction'] = predictions
    entry_points = df[df['Prediction'] == 1]
    return entry_points

def process_ticker(ticker):
    try:
        df = download_data(ticker)
        df = calculate_indicators(df)
        df = create_labels(df)
        features, target, _ = prepare_data(df)
        
        if len(np.unique(target)) < 2:
            raise ValueError(f"The data for {ticker} contains only one class. Ensure that your data has both positive and negative examples.")
        
        features_resampled, target_resampled = resample_data(features, target)
        X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print(f"\nClassification Report for {ticker}:\n")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        entry_points = find_entry_points(df, model)
        
        print(f"\nPotential Entry Points for {ticker}:\n")
        print(entry_points[['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_high', 'BB_low', 'BB_mavg', 'SMA', 'EMA', 'ATR', 'Stoch', 'Stoch_signal', 'OBV']])

    except Exception as e:
        print(f"An error occurred for {ticker}: {e}")

def main():
    tickers = ['AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X','NZDUSD=X','USDCHF=X','USDCAD=X']
    for ticker in tickers:
        process_ticker(ticker)

if __name__ == "__main__":
    main()
