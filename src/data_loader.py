"""
Intraday ETF Data Downloader & Cleaner (Alpaca + NYSE Calendar)

This script:
  - Connects to Alpaca’s Market Data API (using API keys stored in .env)
  - Requests 1-minute bar data for chosen symbols
  - Aligns data to the official NYSE calendar (no weekends/holidays)
  - Restricts to regular trading hours (09:30–16:00)
  - Regularizes missing minutes (forward-fills OHLC/VWAP, zero-fills volume/trades)
  - Saves one Parquet file per symbol in data/raw/

How to customize:
  - Edit `symbols` to choose tickers (e.g. ["SPY","QQQ","DIA","IWM"])
  - Edit `start_year` and `end_year` to control how many years of data to fetch
  - Adjust `timeframe` if you want a different granularity (Minute, Hour, Day)
"""

import os
from dotenv import load_dotenv
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas_market_calendars as mcal 

# ----------------------
# 1. Load API keys
# ----------------------
load_dotenv()
api_key = os.getenv("APCA_API_KEY_ID")
api_secret = os.getenv("APCA_API_SECRET_KEY")

client = StockHistoricalDataClient(api_key, api_secret)

# ----------------------
# 2. Function: Regularize with NYSE calendar
# ----------------------
def regularize_symbol(df, symbol, start_date, end_date):
    """
    Regularize a symbol's intraday bars:
      - Build full 1-min grid only on actual NYSE trading days
      - Restrict to RTH (09:30–16:00)
      - Forward-fill OHLC/VWAP, zero-fill volume/trade_count
    """
    # Filter one symbol’s raw bars
    df = df[df['symbol'] == symbol].copy().set_index("timestamp")

    # Get actual NYSE trading days between start/end
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start_date.date(), end_date=end_date.date())
    trading_minutes = mcal.date_range(schedule, frequency="1min")

    # Reindex raw bars to the trading minutes grid
    df = df.reindex(trading_minutes)

    # Forward-fill OHLC + VWAP
    df[['open','high','low','close','vwap']] = df[['open','high','low','close','vwap']].ffill()

    # Zero-fill volume & trade counts
    df[['volume','trade_count']] = df[['volume','trade_count']].fillna(0)

    # Reset index back to column
    df = df.reset_index().rename(columns={'index':'timestamp'})
    df['symbol'] = symbol
    return df

# ----------------------
# 3. Configuration
# ----------------------
symbols = ["SPY", "QQQ"]            # List of tickers to download
start_year = 2020                   # First year of data
end_year   = 2023                   # Last year of data (inclusive)

os.makedirs("data/raw", exist_ok=True)

# ----------------------
# 4. Loop over years and symbols
# ----------------------
for year in range(start_year, end_year+1):
    start_date = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    end_date   = pd.Timestamp(f"{year}-12-31", tz="America/New_York")

    print(f"\nFetching {year} data...")
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )
    bars = client.get_stock_bars(request_params).df.reset_index()

    # Apply per symbol and save
    for sym in symbols:
        clean_df = regularize_symbol(bars, sym, start_date, end_date)
        outpath = f"data/raw/{sym}_1min_{year}.parquet"
        clean_df.to_parquet(outpath, index=False)
        print(f"  Saved {sym}: {len(clean_df)} rows → {outpath}")
