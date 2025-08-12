import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

# === CONFIG ===
symbol = "AAPL"            # Change this to your desired ticker
start_date = "2022-01-01"  # Start date for data
end_date = "2025-08-11"    # End date for data

print(f"Downloading {symbol} data from {start_date} to {end_date}...")

# === Download Data ===
df = yf.download(symbol, start=start_date, end=end_date, interval="1d")

# Handle MultiIndex columns (happens if yfinance returns multiple tickers or adjusted data)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Locate the correct OHLCV columns
ohlc_cols = [col for col in df.columns if any(x in col for x in ["Open", "High", "Low", "Close"])]
volume_col = [col for col in df.columns if "Volume" in col][0]

# Ensure Volume is numeric
df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce').fillna(0)

# Rename columns for mplfinance compatibility
rename_map = {
    [c for c in df.columns if "Open" in c][0]: "Open",
    [c for c in df.columns if "High" in c][0]: "High",
    [c for c in df.columns if "Low" in c][0]: "Low",
    [c for c in df.columns if "Close" in c][0]: "Close",
    volume_col: "Volume"
}
df.rename(columns=rename_map, inplace=True)

# Keep only required columns
df = df[["Open", "High", "Low", "Close", "Volume"]]

# === Plot Candlestick Chart ===
mpf.plot(
    df,
    type='candle',
    volume=True,
    style='yahoo',
    title=f"{symbol} Candlestick Chart",
    ylabel="Price",
    ylabel_lower="Volume",
    figratio=(14,7),
    figscale=1.2
)

print("✅ Done!")
# === End of Script ===