# 📈 History.py

A simple Python utility for calculating **annual percentage returns** of stocks, ETFs, or indices using **Yahoo Finance** data.  
It supports adjusted/total return calculations, custom fiscal year alignment, and flexible start/end dates.

---

## 🚀 Features

- Fetches price history via [yfinance](https://pypi.org/project/yfinance/).
- Calculates **annual returns** as a `pandas.Series`.
- Option to use **total return** (adjusted close) or **price return** (close).
- Supports **custom fiscal year ends** (e.g., March, June, September, etc.).
- Simple, extensible, and built on standard Python libraries.

---

## 📦 Installation

Clone the repository and install required dependencies:

### Requirements
- Python 3.8+
- pandas
- yfinance
- scipy

---

## 🔧 Usage

```python
import history

# Example: Get annual returns for SPY since 2000
returns = history.annual_returns(
    ticker="SPY",
    start="2000-01-01",
    use_total_return=True,   # Use adjusted close
    fy_end_month=12          # Fiscal year end (default: December)
)

print(returns)
```

### Sample Output

```
2000   -9.10
2001  -11.89
2002  -22.10
2003   28.68
...
```

---

## ⚙️ Function Parameters

`annual_returns(ticker: str, start: str = "1900-01-01", end: Optional[str] = None, use_total_return: bool = True, fy_end_month: Optional[int] = None) -> pd.Series`

| Parameter         | Type      | Default       | Description |
|-------------------|-----------|---------------|-------------|
| `ticker`          | str       | *required*    | Stock/ETF ticker symbol (e.g., `"AAPL"`, `"SPY"`) |
| `start`           | str       | `"1900-01-01"` | Start date for historical data |
| `end`             | str/None  | None          | End date (defaults to today) |
| `use_total_return`| bool      | True          | Use adjusted close for total return |
| `fy_end_month`    | int/None  | None          | Fiscal year end month (1=Jan, 12=Dec). If None, defaults to calendar year |

---

## 🧪 Example: Non-December Fiscal Year

```python
# Get annual returns for Microsoft with fiscal year ending in June
returns = history.annual_returns(
    ticker="MSFT",
    start="2010-01-01",
    fy_end_month=6
)

print(returns)
```

---

## 🛠️ Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you’d like to change.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
