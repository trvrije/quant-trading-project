# Quant Trading Project
This project tests intraday equities/ETF strategies using Alpaca API data.

1. Data gaps arise because Alpaca’s free plan only includes IEX trades. Missing minutes were forward-filled for OHLC/VWAP and volume was set to zero. This creates a regular grid needed for backtesting, but may understate volatility in thinly traded ETFs.