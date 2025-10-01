import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# Initialize client
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Request 1-minute bars for SPY in January 2023
request_params = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 1, 31)
)

bars = client.get_stock_bars(request_params)

# Convert to DataFrame
df = bars.df
print(df.head())

# Save locally
df.to_parquet("data/raw/SPY_1min_Jan2023.parquet")
print("Saved SPY 1-min bars to data/raw/")
