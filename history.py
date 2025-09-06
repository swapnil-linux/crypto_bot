# history.py
import sys
import pandas as pd
import yfinance as yf
from typing import Optional
from scipy import stats   # pip install scipy

def annual_returns(
    ticker: str,
    start: str = "1900-01-01",
    end: Optional[str] = None,
    use_total_return: bool = True,
    fy_end_month: Optional[int] = None
) -> pd.Series:
    """
    Returns a pd.Series of annual percentage returns for the given ticker.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=True)
    if df.empty:
        raise ValueError(f"No data found for {ticker} in the requested range.")

    if use_total_return and "Adj Close" in df.columns:
        px = df["Adj Close"].dropna()
    else:
        px = df["Close"].dropna()

    if fy_end_month is None:
        period_last = px.resample("YE-DEC").last()
    else:
        month_name = pd.to_datetime(f"2000-{fy_end_month:02d}-01").strftime("%b").upper()
        period_last = px.resample(f"YE-{month_name}").last()

    ann_ret = period_last.pct_change().dropna() * 100.0
    ann_ret.index = ann_ret.index.year
    ann_ret.name = f"{ticker} Annual Return (%)"
    return ann_ret.round(2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python history.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    pd.set_option("display.max_rows", None)

    returns = annual_returns(ticker, start="1900-01-01", use_total_return=True)
    print(returns)

    # Save to CSV
    returns.to_csv(f"{ticker}_annual_returns.csv", header=["Annual %"])
    print(f"\nSaved results to {ticker}_annual_returns.csv")
    
    # Ensure we have a Series of values
    if isinstance(returns, pd.DataFrame):
        values = returns.iloc[:, 0]   # take first column if DataFrame
    else:
        values = returns

    # Summary stats
    mean_val = values.mean()
    median_val = values.median()
    sd_val = values.std()

    # One-sample t-test against 0
    t_stat, p_val = stats.ttest_1samp(values, 0)

    print("\n📊 Summary Statistics:")
    print(f"Mean   : {mean_val:.2f}%")
    print(f"Median : {median_val:.2f}%")
    print(f"Std Dev: {sd_val:.2f}%")
    print(f"T-stat : {t_stat:.3f}")
    print(f"P-value: {p_val:.5f}")

    if p_val < 0.05:
        print("✅ Statistically significant: mean return is different from 0 (95% confidence).")
    else:
        print("⚠️ Not statistically significant: cannot reject the null hypothesis (mean = 0).")

