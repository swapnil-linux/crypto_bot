# Forex Trading Strategy with Machine Learning

This project implements a forex trading strategy using various technical indicators and machine learning to identify potential entry points. The key components include data download, calculation of technical indicators, label creation, data preparation, resampling, model training, and prediction of entry points.

## Requirements

To run this project, you need to have the following Python packages installed:

- yfinance
- pandas
- numpy
- ta
- scikit-learn
- imbalanced-learn

You can install these packages using pip:

```bash
pip install yfinance pandas numpy ta scikit-learn imbalanced-learn
```

## Project Structure

The project consists of the following main functions:

1. **download_data**: Downloads historical forex data.
2. **calculate_additional_indicators**: Calculates additional technical indicators.
3. **calculate_indicators**: Calculates primary technical indicators.
4. **create_labels**: Creates labels for training based on future price movements.
5. **prepare_data**: Prepares data for model training.
6. **resample_data**: Resamples data to address class imbalance.
7. **train_model**: Trains a Random Forest classifier.
8. **find_entry_points**: Finds potential entry points based on model predictions.
9. **process_ticker**: Processes each forex ticker symbol.
10. **main**: Main function to run the entire process.

## Functions Overview

### download_data

```python
def download_data(ticker='AUDUSD=X', interval='5m', period='1mo'):
    df = yf.download(ticker, interval=interval, period=period)
    df['Volume'] = df['Volume'].replace(to_replace=0, method='ffill')
    return df
```

Downloads historical forex data for a given ticker symbol, interval, and period.

### calculate_additional_indicators

```python
def calculate_additional_indicators(df):
    df['SMA'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch'] = stoch.stoch()
    df['Stoch_signal'] = stoch.stoch_signal()
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    return df
```

Calculates additional technical indicators such as SMA, EMA, ATR, Stochastic Oscillator, and OBV.

### calculate_indicators

```python
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
```

Calculates primary technical indicators such as RSI, MACD, and Bollinger Bands.

### create_labels

```python
def create_labels(df, take_profit=0.0005):
    df['Future_Close'] = df['Close'].shift(-1)
    df['Price_Change'] = df['Future_Close'] - df['Close']
    df['Target'] = np.where(df['Price_Change'] > take_profit, 1, 0)
    return df
```

Creates labels for training based on future price changes.

### prepare_data

```python
def prepare_data(df):
    df = df.dropna()
    features = df[['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_high', 'BB_low', 'BB_mavg', 'SMA', 'EMA', 'ATR', 'Stoch', 'Stoch_signal', 'OBV']]
    target = df['Target']
    return features, target, df.index
```

Prepares data for model training by selecting features and target variables.

### resample_data

```python
def resample_data(features, target):
    smote = SMOTE()
    features_resampled, target_resampled = smote.fit_resample(features, target)
    return features_resampled, target_resampled
```

Resamples data to address class imbalance using SMOTE.

### train_model

```python
def train_model(features, target):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    return model
```

Trains a Random Forest classifier.

### find_entry_points

```python
def find_entry_points(df, model):
    features, _, indices = prepare_data(df)
    predictions = model.predict(features)
    df.loc[indices, 'Prediction'] = predictions
    entry_points = df[df['Prediction'] == 1]
    return entry_points
```

Finds potential entry points based on model predictions.

### process_ticker

```python
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
```

Processes each forex ticker symbol by downloading data, calculating indicators, creating labels, training the model, and finding potential entry points.

### main

```python
def main():
    tickers = ['AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X','NZDUSD=X','USDCHF=X','USDCAD=X']
    for ticker in tickers:
        process_ticker(ticker)

if __name__ == "__main__":
    main()
```

Main function to run the entire process for a list of forex ticker symbols.

## How to Use

1. Clone the repository.
2. Install the required packages.
3. Run the script:

```bash
python script_name.py
```

The script will download historical forex data, calculate technical indicators, create labels, train a Random Forest classifier, and print potential entry points for each ticker symbol.

## License

This project is licensed under the MIT License.
