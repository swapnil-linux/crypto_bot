# Trading Signal Generator

This Python script generates trading signals for a list of tickers based on Bollinger Bands, Keltner Channels, and momentum oscillator indicators. The script reads ticker values from a file and processes the data for each ticker to identify buy signals. The tickers and buy signals are displayed in a tabular format.

## Prerequisites

- Python 3.x
- pandas
- numpy
- ta
- yfinance
- prettytable
- tqdm

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/swapnil-linux/crypto-bot.git
   ```

2. Navigate to the project directory:

   ```shell
   cd crypto-bot/TTM
   ```

3. Install the required packages:

   ```shell
   pip install pandas numpy ta yfinance prettytable tqdm
   ```

## Usage

Run the script using the following command:

```shell
python ttm.py [--all] [--snp500]
```

- `--all`: Reads ticker values from the `nasdaq_all.txt` file.
- `--snp500`: Reads ticker values from the `snp500.txt` file.
- If no arguments are provided, it defaults to reading ticker values from the `snp500.txt` file.

The script processes the data for each ticker and displays the tickers along with the last price and trading signals in a tabular format.

## Sample Output

```
+--------+------------+------+
| Ticker | Last price | LONG |
+--------+------------+------+
|  AAPL  |   145.64   | LONG |
|  TSLA  |   718.28   | LONG |
|  GOOG  |  2704.42   | LONG |
+--------+------------+------+
```

## License

This project is licensed under the [MIT License](LICENSE).
```