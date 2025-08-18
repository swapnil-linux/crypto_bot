#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TTM Squeeze-style breakout scanner.

Requirements:
  pip install pandas numpy yfinance ta prettytable tqdm
"""

import os
import argparse
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from prettytable import PrettyTable
from tqdm import tqdm
from multiprocessing import Pool

try:
    import ta
except Exception as e:
    raise SystemExit("Install 'ta' with: pip install ta\n" + str(e))


# -----------------------------
# Parameters bag
# -----------------------------
@dataclass(frozen=True)
class Params:
    length: int = 20
    bb_mult: float = 2.0
    kc_mult_high: float = 1.0
    kc_mult_mid: float = 1.5
    kc_mult_low: float = 2.0
    squeeze_bars: int = 7
    auto_adjust: bool = True


PARAMS = Params()
VERBOSE = False


def _init_pool(params: Params, verbose: bool):
    global PARAMS, VERBOSE
    PARAMS = params
    VERBOSE = verbose


def safe_base_dir() -> str:
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


def compute_bollinger(close: pd.Series, length: int, mult: float):
    basis = close.rolling(length).mean()
    dev = mult * close.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    return basis, upper, lower


def compute_keltner(df: pd.DataFrame, length: int,
                    mult_high: float, mult_mid: float, mult_low: float):
    basis = df["Close"].ewm(span=length, adjust=False).mean()
    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=length
    ).average_true_range()
    return {
        "basis": basis,
        "upper_high": basis + atr * mult_high,
        "lower_high": basis - atr * mult_high,
        "upper_mid":  basis + atr * mult_mid,
        "lower_mid":  basis - atr * mult_mid,
        "upper_low":  basis + atr * mult_low,
        "lower_low":  basis - atr * mult_low,
    }


def linear_reg_slope(series: pd.Series, length: int) -> pd.Series:
    idx = np.arange(length)

    def _slope(window: np.ndarray) -> float:
        return float(np.polyfit(idx, window, 1)[0])

    return series.rolling(length).apply(_slope, raw=True)


def process_ticker(ticker: str):
    try:
        p = PARAMS
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=p.auto_adjust,
        )

        min_len = max(p.length + 21, p.length + p.squeeze_bars + 2)
        if df is None or df.empty or len(df) < min_len:
            return None

        _, bb_upper, bb_lower = compute_bollinger(df["Close"], p.length, p.bb_mult)
        kc = compute_keltner(df, p.length, p.kc_mult_high, p.kc_mult_mid, p.kc_mult_low)

        low_sqz = (bb_lower >= kc["lower_low"]) & (bb_upper <= kc["upper_low"])
        mid_sqz = (bb_lower >= kc["lower_mid"]) & (bb_upper <= kc["upper_mid"])
        high_sqz = (bb_lower >= kc["lower_high"]) & (bb_upper <= kc["upper_high"])
        df["in_squeeze"] = (low_sqz | mid_sqz | high_sqz).astype(int)

        avg = (df["High"].rolling(p.length).max() + df["Low"].rolling(p.length).min()) / 2.0
        mom_input = df["Close"] - avg
        mom = linear_reg_slope(mom_input, p.length)

        ema8 = df["Close"].ewm(span=8, adjust=False).mean()
        ema21 = df["Close"].ewm(span=21, adjust=False).mean()

        prior_all_sqz = df["in_squeeze"].shift(1).rolling(p.squeeze_bars, min_periods=p.squeeze_bars).min() == 1
        just_exited = (df["in_squeeze"].shift(1) == 1) & (df["in_squeeze"] == 0)

        df["buy_signal"] = prior_all_sqz & just_exited & (mom > 0) & (df["Close"] > ema8) & (ema8 > ema21)

        if bool(df["buy_signal"].iloc[-1]):
            last_price = round(float(df["Close"].iloc[-1]), 2)
            return [ticker, last_price, "LONG"]

        if VERBOSE:
            debug = {
                "prior_all_sqz": bool(prior_all_sqz.iloc[-1]),
                "just_exited": bool(just_exited.iloc[-1]),
                "mom>0": bool((mom > 0).iloc[-1]),
                "close>ema8": bool((df["Close"] > ema8).iloc[-1]),
                "ema8>ema21": bool((ema8 > ema21).iloc[-1]),
                "in_squeeze": int(df["in_squeeze"].iloc[-1]),
            }
            print(f"{ticker}: NO SIGNAL -> {debug}")

        return None

    except Exception as e:
        return [ticker, f"Error: {str(e)}", "ERROR"]


def main():
    parser = argparse.ArgumentParser(description="Squeeze breakout scanner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Read from nasdaq_all.txt")
    group.add_argument("--snp500", action="store_true", help="Read from snp500.txt")
    group.add_argument("--nifty50", action="store_true", help="Read from nifty50.txt")
    parser.add_argument("--file", help="Custom ticker file path")

    parser.add_argument("--length", type=int, default=Params.length, help="Lookback length (default: 20)")
    parser.add_argument("--bb-mult", type=float, default=Params.bb_mult, help="Bollinger multiplier (default: 2.0)")
    parser.add_argument("--kc-high", type=float, default=Params.kc_mult_high, help="Keltner high multiplier (default: 1.0)")
    parser.add_argument("--kc-mid", type=float, default=Params.kc_mult_mid, help="Keltner mid multiplier (default: 1.5)")
    parser.add_argument("--kc-low", type=float, default=Params.kc_mult_low, help="Keltner low multiplier (default: 2.0)")
    parser.add_argument("--squeeze-bars", type=int, default=Params.squeeze_bars, help="Bars in squeeze before breakout (default: 7)")
    parser.add_argument("--auto-adjust", type=lambda s: s.lower() == "true", default="True",
                        help="yfinance auto_adjust (True/False, default: True)")
    parser.add_argument("--max-workers", type=int, default=8, help="Max parallel workers (default: 8)")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics even when no signal")

    args = parser.parse_args()

    if args.file:
        file_name = args.file
    elif args.all:
        file_name = "nasdaq_all.txt"
    elif args.snp500:
        file_name = "snp500.txt"
    elif args.nifty50:
        file_name = "nifty50.txt"
    else:
        file_name = "snp500.txt"

    params = Params(
        length=args.length,
        bb_mult=args.bb_mult,
        kc_mult_high=args.kc_high,
        kc_mult_mid=args.kc_mid,
        kc_mult_low=args.kc_low,
        squeeze_bars=args.squeeze_bars,
        auto_adjust=(args.auto_adjust if isinstance(args.auto_adjust, bool) else args.auto_adjust.lower() == "true"),
    )

    cpu = os.cpu_count() or 1
    max_workers = max(1, min(args.max_workers, 8, cpu))

    tickers = read_tickers(file_name)

    table = PrettyTable()
    table.field_names = ["Ticker", "Last Price", "Signal"]

    with Pool(processes=max_workers, initializer=_init_pool, initargs=(params, args.verbose)) as p:
        for result in tqdm(
            p.imap_unordered(process_ticker, tickers),
            total=len(tickers),
            unit="ticker",
            desc="Scanning",
        ):
            if result and result[2] == "LONG":
                table.add_row(result)

    print(table)


if __name__ == "__main__":
    main()
