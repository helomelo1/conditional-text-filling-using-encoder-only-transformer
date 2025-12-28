import yfinance as yf
import numpy as np
from pathlib import Path


tickers = ["^GSPC", "BTC-USD"]

def download_data(ticker="^GSPC", start="2020-01-01", end=None, save_path=""):
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)

        prices = data[['Close']].dropna()
        save_path = f"{ticker}.csv"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(save_path)

        print(f"Saved {len(prices)} rows to {save_path}")

if __name__ == "__main__":
    download_data()