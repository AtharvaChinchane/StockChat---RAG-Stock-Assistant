import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Settings ---
tickers = [
    # Indian companies (~70)
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "ITC.NS", "MARUTI.NS", "AXISBANK.NS", "HCLTECH.NS",
    "BAJAJFINSV.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "ONGC.NS", "IOC.NS", "VEDL.NS", "GRASIM.NS", "HDFCLIFE.NS", "TATAMOTORS.NS",
    "INDUSINDBK.NS", "TECHM.NS", "SUNPHARMA.NS", "BRITANNIA.NS", "WIPRO.NS", "CIPLA.NS", "BPCL.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "ICICIPRULI.NS", "BHARATFORG.NS",
    "GAIL.NS", "NMDC.NS", "SHREECEM.NS", "ABB.NS", "SBILIFE.NS", "ADANIGREEN.NS", "BANKBARODA.NS",
    "PETRONET.NS", "COLPAL.NS", "TORNTPHARM.NS", "ISEC.NS", "DABUR.NS", "MARICO.NS", "PIDILITIND.NS",
    "MOTHERSUMI.NS", "VOLTAS.NS", "INDIGO.NS", "BAJAJHLDNG.NS", "LICHSGFIN.NS", "TATAPOWER.NS",
    "HAVELLS.NS", "ACC.NS", "BOSCHLTD.NS", "PVR.NS", "ZEEL.NS",

    # US & Global companies (~30)
    "AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "NFLX", "INTC", "AMD",
    "ADBE", "PYPL", "CSCO", "ORCL", "IBM", "BA", "DIS", "V", "MA", "KO",
    "GM", "F", "PEP", "MCD", "WMT", "JNJ", "PG", "XOM", "CVX", "PFE"
]

data_dir = "stock_data"
os.makedirs(data_dir, exist_ok=True)

today = datetime.today().date()

for ticker in tickers:
    csv_path = os.path.join(data_dir, f"{ticker}.csv")

    # Determine start date: last saved date + 1 day, else default to today
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if not df_existing.empty:
            start_date = df_existing.index[-1].date() + timedelta(days=1)
        else:
            start_date = today
    else:
        start_date = today

    if start_date > today:
        print(f"⚠️ No new data to fetch for {ticker}")
        continue

    try:
        # Use Ticker.history() to always get a DataFrame
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=today + timedelta(days=1), auto_adjust=False)

        if data.empty:
            print(f"⚠️ No new data for {ticker}")
            continue

        # Reset index if not already
        if isinstance(data.index, pd.DatetimeIndex):
            data.index.name = "Date"

        # Append to existing CSV or create new
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            data = pd.concat([df_existing, data])
            data = data[~data.index.duplicated(keep='last')]  # Remove duplicates

        data.to_csv(csv_path)
        print(f"✅ Updated data for {ticker} ({len(data)} rows)")

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
