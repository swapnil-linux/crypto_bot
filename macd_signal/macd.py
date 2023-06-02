import yfinance as yf
from pycoingecko import CoinGeckoAPI
from prettytable import PrettyTable
from ta.trend import MACD
import numpy as np
from tqdm import tqdm

cg = CoinGeckoAPI()

def get_top_crypto_symbols(n):
    top_20_cryptos = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=n)
    symbols = [crypto['symbol'].upper() + '-USD' for crypto in top_20_cryptos]
    return symbols


# Define list of tickers
tickers = ['AAPL', 'ABBV', 'ABNB', 'ABT', 'ACN', 'ADANIPORTS.NS', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALGN', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMZN', 'ANSS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'ASML', 'ATVI', 'AVGO', 'AXISBANK.NS', 'AXP', 'AZN', 'BA', 'BAC', 'BAJAJFINSV.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BIIB', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BPCL.NS', 'BRITANNIA.NS', 'BRK-B', 'C', 'CAT', 'CDNS', 'CEG', 'CHTR', 'CIPLA.NS', 'CL', 'CMCSA', 'COALINDIA.NS', 'COF', 'COP', 'COST', 'CPRT', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CVS', 'CVX', 'DDOG', 'DHR', 'DIS', 'DIVISLAB.NS', 'DLTR', 'DOW', 'DRREDDY.NS', 'DUK', 'DXCM', 'EA', 'EBAY', 'EICHERMOT.NS', 'EMR', 'ENPH', 'EXC', 'F', 'FANG', 'FAST', 'FDX', 'FISV', 'FTNT', 'GD', 'GE', 'GFS', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GRASIM.NS', 'GS', 'HCLTECH.NS', 'HD', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'HON', 'IBM', 'ICICIBANK.NS', 'IDXX', 'ILMN', 'INDUSINDBK.NS', 'INFY.NS', 'INTC', 'INTU', 'ISRG', 'ITC.NS', 'JD', 'JNJ', 'JPM', 'JSWSTEEL.NS', 'KDP', 'KHC', 'KLAC', 'KO', 'KOTAKBANK.NS', 'LCID', 'LIN', 'LLY', 'LMT', 'LOW', 'LRCX', 'LT.NS', 'LULU', 'MA', 'MAR', 'MARUTI.NS', 'MCD', 'MCHP', 'MDLZ', 'MDT', 'MELI', 'META', 'MMM', 'MNST', 'MO', 'MRK', 'MRNA', 'MRVL', 'MS', 'MSFT', 'MU', 'M&M.NS', 'NEE', 'NESTLEIND.NS', 'NFLX', 'NKE', 'NTPC.NS', 'NVDA', 'NXPI', 'ODFL', 'ONGC.NS', 'ORCL', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PFE', 'PG', 'PM', 'POWERGRID.NS', 'PYPL', 'QCOM', 'REGN', 'RELIANCE.NS', 'RIVN', 'ROST', 'RTX', 'SBILIFE.NS', 'SBIN.NS', 'SBUX', 'SCHW', 'SGEN', 'SIRI', 'SNPS', 'SO', 'SPG', 'SUNPHARMA.NS', 'T', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TEAM', 'TECHM.NS', 'TGT', 'TITAN.NS', 'TMO', 'TMUS', 'TSLA', 'TXN', 'ULTRACEMCO.NS', 'UNH', 'UNP', 'UPL.NS', 'UPS', 'USB', 'V', 'VRSK', 'VRTX', 'VZ', 'WBA', 'WBD', 'WDAY', 'WFC', 'WIPRO.NS', 'WMT', 'XEL', 'XOM', 'ZM', 'ZS','UI','SEDG','LOVE']
#tickers = ['AAPL']
#cryptos = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'MATIC-USD', 'DOT-USD', 'SOL-USD', 'LTC-USD', 'SHIB-USD', 'CRO-USD', 'VET-USD', 'FTM-USD', 'BAKE-USD', 'WAVES-USD', 'AGLD-USD', 'CHR-USD' ]
#cryptos = ['BTC-USD']
# Add top 20 crypto symbols to the stock_tickers list
cryptos = get_top_crypto_symbols(20)
tickers.extend(cryptos)

# Initialize the table
table = PrettyTable()
table.field_names = ["Ticker", "Last price","LONG"]

table_sell = PrettyTable()
table_sell.field_names = ["Ticker", "Last price","SHORT"]

table_err = PrettyTable()
table_err.field_names = ["Ticker", "Error"]

# Loop over tickers 

with tqdm(total=len(tickers), unit="ticker") as pbar:
    for i, ticker in enumerate(tickers):
        pbar.set_description(f"Downloading data for {ticker}")

        try:
            data = yf.download(ticker, period="1y", interval="1d", progress=False)
            macd = MACD(data['Close'])
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            # BUY SIGNAL
            macd_cross_up_signal = np.where((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 1, 0)
            macd_below_zero = np.where((macd_cross_up_signal == 1) & (macd_line < 0), 1, 0)

            # SELL SIGNAL
            macd_cross_down_signal = np.where((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), 1, 0)
            macd_above_zero = np.where((macd_cross_down_signal == 1) & (macd_line > 0), 1, 0)

            last_price = round(data['Close'][-1],2)
            if any(macd_below_zero[-3:] == 1):
                table.add_row([ticker, last_price,"LONG"])

            if any(macd_above_zero[-3:] == 1):
                table_sell.add_row([ticker, last_price,"SHORT"])

            pbar.update(1)   
        except KeyError:
            table_err.add_row([ticker, "Data not found"])
        except IndexError:
            table_err.add_row([ticker, "Empty DataFrame"])
        except Exception as e:
            table_err.add_row([ticker, f"Error: {str(e)}"])
print(table)
print(table_sell)
#print(table_err)
