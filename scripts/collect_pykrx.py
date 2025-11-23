from pykrx import stock
import pandas as pd
from datetime import datetime

def get_kospi(start="20180101", end=None):
    if end is None:
        end = datetime.today().strftime("%Y%m%d")

    print(f"[INFO] 코스피모으는중 data from {start} to {end}")

    df = stock.get_index_ohlcv(start, end, "1001")  # KOSPI 지수 코드 1001
    df["return"] = df["종가"].pct_change()
    df["volatility"] = df["return"].rolling(5).std()

    return df

if __name__ == "__main__":
    df = get_kospi()
    df.to_csv("../data/raw/kospi.csv", encoding="utf-8-sig")
    print("[SUCCESS] Saved to data/raw/kospi.csv")
