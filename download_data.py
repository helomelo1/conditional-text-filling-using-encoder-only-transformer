import yfinance as yf
import numpy as np
from pathlib import Path


def download_data(ticker, start, end, save_path="."):
    data = yf.download(ticker, start=start, end=end, progress=False)

    prices = data[['Close']].dropna()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(save_path)

    print(f"Saved {len(prices)} rows to {save_path}")

if __name__ == "__main__":
    download_data()