#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Squeeze scanner with Bollinger Bands + Keltner Channels and breakout filter.

Requirements:
  pip install pandas numpy yfinance ta prettytable tqdm

Notes:
- Uses 'ta' (technical-analysis) library, not pandas-ta.
- ATR via ta.volatility.AverageTrueRange.
- Keltner basis uses EMA; Bollinger uses population std (ddof=0).
"""

import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from prettytable import PrettyTable
from tqdm import tqdm
from multiprocessing import Pool
import warnings

# Optional: quiet down numpy polyfit RankWarning on flat windows
warnings.simplefilter("ignore", np.RankWarning)

# -----------------------------
# Defaults (overridable by CLI)
# -----------------------------
LENGTH = 20
BB_MULT = 2.0
KC_MULT_HIGH = 1.0
KC_MULT_MID = 1.5
KC_MULT_LOW = 2.0
SQUEEZE_BARS = 7
MAX_WORKERS = 8

# Lazy import so the script prints a nicer message if 'ta' is missing
try:
    import ta
except Exception as e:
    raise SystemExit(
        "The 'ta' library is required. Install with:\n  pip install ta\n\n"
        f"Import error: {e}"
    )


def safe_base_dir() -> str:
    """Handle __file__ not existing in some environments."""
    return (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals()
        else os.getcwd()
    )


def read_tickers(file_name: str) -> list[str]:
    base_dir = safe_base_dir()
    file_path = os.path.join(base_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ticker file not found: {file_path}")
    with open(file_path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


def compute_bollinger(close: pd.Series, length: int, mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = close.rolling(length).mean()
    dev = mult * close.rolling(length).std(ddof=0)  # population std for consistency
    upper = basis + dev
    lower = basis - dev
    return basis, upper, lower


def compute_keltner(df: pd.DataFrame, length: int,
                    mult_high: float, mult_mid: float, mult_low: float) -> dict:
    # Common definitions use an EMA for the central line
    basis = df["Close"].ewm(span=length, adjust=False).mean()

    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=length
    ).average_true_range()

    levels = {
        "basis": basis,
        "upper_high": basis + atr * mult_high,
        "lower_high": basis - atr * mult_high,
        "upper_mid":  basis + atr * mult_mid,
        "lower_mid":  basis - atr * mult_mid,
        "upper_low":  basis + atr * mult_low,
        "lower_low":  basis - atr * mult_low,
    }
    return levels


def linear_reg_slope(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling linear regression slope using numpy.polyfit on a sliding window.
    Slope of (series) over window indexes 0..length-1.
    """
    idx = np.arange(length)

    def _slope(window: np.ndarray) -> float:
        # window is a numpy array, length == length
        # polyfit returns slope, intercept; we want slope
        return float(np.polyfit(idx, window, 1)[0])

    return series.rolling(length).apply(_slope, raw=True)


def process_ticker(ticker: str):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)

        # Basic sanity: need enough bars for indicators and EMA stack
        min_len = max(LENGTH + 21, LENGTH + SQUEEZE_BARS + 2)
        if df is None or df.empty or len(df) < min_len:
            return None

        # Bollinger
        bb_basis, bb_upper, bb_lower = compute_bollinger(df["Close"], LENGTH, BB_MULT)

        # Keltner
        kc = compute_keltner(
            df,
            length=LENGTH,
            mult_high=KC_MULT_HIGH,
            mult_mid=KC_MULT_MID,
            mult_low=KC_MULT_LOW,
        )

        # Squeeze conditions (three compression tiers)
        low_sqz = (bb_lower >= kc["lower_low"]) & (bb_upper <= kc["upper_low"])
        mid_sqz = (bb_lower >= kc["lower_mid"]) & (bb_upper <= kc["upper_mid"])
        high_sqz = (bb_lower >= kc["lower_high"]) & (bb_upper <= kc["upper_high"])

        df["in_squeeze"] = (low_sqz | mid_sqz | high_sqz).astype(int)

        # Momentum proxy: slope of (close - mid of rolling high/low)
        avg = (df["High"].rolling(LENGTH).max() + df["Low"].rolling(LENGTH).min()) / 2.0
        mom_input = df["Close"] - avg
        mom = linear_reg_slope(mom_input, LENGTH)

        # Trend filter
        ema8 = df["Close"].ewm(span=8, adjust=False).mean()
        ema21 = df["Close"].ewm(span=21, adjust=False).mean()

        # "Prior N bars all squeeze" and "just exited today"
        prior_all_sqz = df["in_squeeze"].shift(1).rolling(SQUEEZE_BARS, min_periods=SQUEEZE_BARS).min() == 1
        just_exited = (df["in_squeeze"].shift(1) == 1) & (df["in_squeeze"] == 0)

        df["buy_signal"] = (
            prior_all_sqz
            & just_exited
            & (mom > 0)
            & (df["Close"] > ema8)
            & (ema8 > ema21)
        )

        if bool(df["buy_signal"].iloc[-1]):
            last_price = round(float(df["Close"].iloc[-1]), 2)
            return [ticker, last_price, "LONG"]
        return None

    except Exception as e:
        # Return an error row so failures are visible instead of silently dropped
        return [ticker, f"Error: {str(e)}", "ERROR"]


def main():
    parser = argparse.ArgumentParser(description="Squeeze breakout scanner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Read from nasdaq_all.txt")
    group.add_argument("--snp500", action="store_true", help="Read from snp500.txt")
    group.add_argument("--nifty50", action="store_true", help="Read from nifty50.txt")
    parser.add_argument("--file", help="Custom ticker file path")
    parser.add_argument("--length", type=int, default=LENGTH, help="Lookback length (default: 20)")
    parser.add_argument("--bb-mult", type=float, default=BB_MULT, help="Bollinger multiplier (default: 2.0)")
    parser.add_argument("--kc-high", type=float, default=KC_MULT_HIGH, help="Keltner high multiplier (default: 1.0)")
    parser.add_argumen_
