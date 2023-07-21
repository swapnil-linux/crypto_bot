import os
import argparse
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from prettytable import PrettyTable
from tqdm import tqdm
from multiprocessing import Pool

length = 20
BB_mult = 2.0
KC_mult_high = 1.0
KC_mult_mid = 1.5
KC_mult_low = 2.0

def process_ticker(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)

        # BOLLINGER BANDS
        BB_basis = df['Close'].rolling(window=length).mean()
        dev = BB_mult * df['Close'].rolling(window=length).std()
        BB_upper = BB_basis + dev
        BB_lower = BB_basis - dev

        # KELTNER CHANNELS
        KC_basis = df['Close'].rolling(window=length).mean()
        devKC = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], length)
        KC_upper_high = KC_basis + devKC * KC_mult_high
        KC_lower_high = KC_basis - devKC * KC_mult_high
        KC_upper_mid = KC_basis + devKC * KC_mult_mid
        KC_lower_mid = KC_basis - devKC * KC_mult_mid
        KC_upper_low = KC_basis + devKC * KC_mult_low
        KC_lower_low = KC_basis - devKC * KC_mult_low

        # SQUEEZE CONDITIONS
        NoSqz = (BB_lower < KC_lower_low) | (BB_upper > KC_upper_low)  # NO SQUEEZE
        LowSqz = (BB_lower >= KC_lower_low) & (BB_upper <= KC_upper_low)  # LOW COMPRESSION
        MidSqz = (BB_lower >= KC_lower_mid) & (BB_upper <= KC_upper_mid)  # MID COMPRESSION
        HighSqz = (BB_lower >= KC_lower_high) & (BB_upper <= KC_upper_high)  # HIGH COMPRESSION

        # MOMENTUM OSCILLATOR
        avg = (df['High'].rolling(window=length).max() + df['Low'].rolling(window=length).min()) / 2
        mom = (df['Close'] - avg).rolling(window=length).apply(lambda x: np.polyfit(range(length), x, 1)[0], raw=False)

        # get last price
        last_price = round(df['Close'][-1], 2)

        # DEFINE BUY SIGNAL
        df['in_squeeze'] = (LowSqz | MidSqz | HighSqz).astype(int)
        df['buy_signal'] = ((df['in_squeeze'].shift(7) == 1) &
                            (df['in_squeeze'].shift().rolling(window=7).min() == 1) &
                            (df['in_squeeze'] == 0) &
                            (mom > 0) &
                            (df['Close'] > df['Close'].ewm(span=8).mean()) &
                            (df['Close'].ewm(span=8).mean() > df['Close'].ewm(span=21).mean()))

        if (df['buy_signal'].iloc[-1] == True):
            return [ticker, last_price, "LONG"]
        else:
            return None
    except Exception as e:
        return [ticker, f"Error: {str(e)}", "N/A"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Read from nasdaq_all.txt')
    parser.add_argument('--snp500', action='store_true', help='Read from snp500.txt')
    args = parser.parse_args()

    if args.all:
        file_name = 'nasdaq_all.txt'
    elif args.snp500:
        file_name = 'snp500.txt'
    else:
        file_name = 'snp500.txt'  # Default to snp500.txt if no arguments provided

    # Get the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create the file path
    file_path = os.path.join(current_directory, file_name)

    tickers = []

    with open(file_path, 'r') as file:
        for line in file:
            ticker = line.strip()  # Remove any leading or trailing whitespace
            tickers.append(ticker)

    # Initialize the table for storing the data.
    table = PrettyTable()
    table.field_names = ["Ticker", "Last price", "LONG"]

    with Pool(os.cpu_count()) as p:
        for result in tqdm(p.imap_unordered(process_ticker, tickers), total=len(tickers), unit="ticker"):
            if result is not None and result[2] == "LONG":  # Only add to table if result is "LONG"
                table.add_row(result)

    print(table)
