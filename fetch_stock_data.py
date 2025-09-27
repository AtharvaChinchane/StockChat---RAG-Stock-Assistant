import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- Settings ---
tickers_map = {
    # Indian companies (~70)
    "Reliance":"RELIANCE.NS", "TCS":"TCS.NS", "HDFC Bank":"HDFCBANK.NS", "Infosys":"INFY.NS",
    "ICICI Bank":"ICICIBANK.NS", "Hindustan Unilever":"HINDUNILVR.NS", "Kotak Mahindra Bank":"KOTAKBANK.NS",
    "Larsen & Toubro":"LT.NS", "State Bank of India":"SBIN.NS", "ITC":"ITC.NS",
    "Maruti Suzuki":"MARUTI.NS", "Axis Bank":"AXISBANK.NS", "HCL Tech":"HCLTECH.NS", 
    "Bajaj Finserv":"BAJAJFINSV.NS", "Bharti Airtel":"BHARTIARTL.NS", "Asian Paints":"ASIANPAINT.NS",
    "Nestle India":"NESTLEIND.NS", "Titan":"TITAN.NS", "UltraTech Cement":"ULTRACEMCO.NS", 
    "Power Grid":"POWERGRID.NS", "ONGC":"ONGC.NS", "IOC":"IOC.NS", "Vedanta":"VEDL.NS",
    "Grasim":"GRASIM.NS", "HDFC Life":"HDFCLIFE.NS", "Tata Motors":"TATAMOTORS.NS", 
    "IndusInd Bank":"INDUSINDBK.NS", "Tech Mahindra":"TECHM.NS", "Sun Pharma":"SUNPHARMA.NS",
    "Britannia":"BRITANNIA.NS", "Wipro":"WIPRO.NS", "Cipla":"CIPLA.NS", "BPCL":"BPCL.NS",
    "Tata Steel":"TATASTEEL.NS", "JSW Steel":"JSWSTEEL.NS", "Adani Ports":"ADANIPORTS.NS",
    "Mahindra & Mahindra":"M&M.NS", "Bajaj Auto":"BAJAJ-AUTO.NS", "Hindalco":"HINDALCO.NS",
    "Divi's Labs":"DIVISLAB.NS", "Dr. Reddy's Labs":"DRREDDY.NS", "Eicher Motors":"EICHERMOT.NS",
    "Hero MotoCorp":"HEROMOTOCO.NS", "ICICI Prudential":"ICICIPRULI.NS", "Bharat Forge":"BHARATFORG.NS",
    "GAIL":"GAIL.NS", "NMDC":"NMDC.NS", "Shree Cement":"SHREECEM.NS", "ABB India":"ABB.NS",
    "SBI Life":"SBILIFE.NS", "Adani Green":"ADANIGREEN.NS", "Bank of Baroda":"BANKBARODA.NS",
    "Petronet LNG":"PETRONET.NS", "Colgate-Palmolive":"COLPAL.NS", "Torrent Pharma":"TORNTPHARM.NS",
    "ICICI Securities":"ISEC.NS", "Dabur":"DABUR.NS", "Marico":"MARICO.NS", "Pidilite":"PIDILITIND.NS",
    "Motherson Sumi":"MOTHERSUMI.NS", "Voltas":"VOLTAS.NS", "Indigo":"INDIGO.NS",
    "Bajaj Holdings":"BAJAJHLDNG.NS", "LIC Housing":"LICHSGFIN.NS", "Tata Power":"TATAPOWER.NS",
    "Havells":"HAVELLS.NS", "ACC":"ACC.NS", "Bosch":"BOSCHLTD.NS", "PVR":"PVR.NS", "Zee Entertainment":"ZEEL.NS",
    
    # US & Global companies (~30)
    "Apple":"AAPL", "Microsoft":"MSFT", "Amazon":"AMZN", "Tesla":"TSLA", "Alphabet":"GOOGL",
    "Facebook":"META", "NVIDIA":"NVDA", "Netflix":"NFLX", "Intel":"INTC", "AMD":"AMD",
    "Adobe":"ADBE", "PayPal":"PYPL", "Cisco":"CSCO", "Oracle":"ORCL", "IBM":"IBM",
    "Boeing":"BA", "Disney":"DIS", "Visa":"V", "Mastercard":"MA", "Coca-Cola":"KO",
    "GM":"GM", "Ford":"F", "Pepsi":"PEP", "McDonald's":"MCD", "Walmart":"WMT",
    "Johnson & Johnson":"JNJ", "Procter & Gamble":"PG", "Exxon Mobil":"XOM", "Chevron":"CVX", "Pfizer":"PFE"
}

# --- Create folders ---
os.makedirs("db", exist_ok=True)
os.makedirs("charts", exist_ok=True)

engine = create_engine("sqlite:///./db/stock_data.db")
today = datetime.today()

for name, ticker in tickers_map.items():
    try:
        print(f"Processing {name} ({ticker})...")
        
        # --- Check last available date in DB ---
        query = f"SELECT MAX(date) as last_date FROM daily_prices WHERE ticker='{ticker}'"
        result = pd.read_sql(query, engine)
        last_date_in_db = result['last_date'][0]
        
        if last_date_in_db:
            start_date = pd.to_datetime(last_date_in_db) + timedelta(days=1)
        else:
            start_date = today - timedelta(days=30)  # fetch last 30 days if no data
        
        if start_date > today:
            print(f"{name} is already up-to-date.")
            continue
        
        # --- Download data ---
        df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))
        if df.empty:
            print(f"No new data for {name}, skipping.")
            continue
        
        df.reset_index(inplace=True)
        df["ticker"] = ticker
        df.rename(columns={
            "Date":"date", "Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"
        }, inplace=True)
        df = df[["ticker","date","open","high","low","close","volume"]]
        df.to_sql("daily_prices", engine, if_exists="append", index=False)
        print(f"{name} data updated in DB.")
        
        # --- Generate chart ---
        plt.figure(figsize=(10,5))
        plt.plot(df['date'], df['close'], marker='o', linestyle='-', color='blue')
        plt.title(f"{name} ({ticker}) Closing Prices")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path = f"charts/{ticker}.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"Chart saved at {chart_path}")
        
    except Exception as e:
        print(f"Error processing {name}: {e}")

print("All done! Database and charts are updated.")
