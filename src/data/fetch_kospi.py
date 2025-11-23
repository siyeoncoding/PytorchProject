# src/data/fetch_kospi.py
import numpy as np
import pandas as pd
from pathlib import Path
from pykrx import stock

from src.config import RAW_KOSPI_DIR, START_DATE, END_DATE


def fetch_kospi_index(start_date, end_date) -> pd.DataFrame:
    """
    KOSPI 지수 데이터를 pykrx로 가져오고
    변동성까지 포함한 최적화된 데이터프레임을 반환.

    반환 컬럼 예:
    ['date', 'close', 'return', 'log_return', 'volatility_10d', 'volume']
    """

    # -------- 1) 데이터 로드 --------
    df = stock.get_index_ohlcv(
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
        "1001"  # KOSPI
    )
    df = df.reset_index().rename(columns={"날짜": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # -------- 2) 기본 피처 생성 --------
    df["close"] = df["종가"]

    df["return"] = df["close"].pct_change().fillna(0.0)

    df["log_return"] = np.log(
        df["close"] / df["close"].shift(1)
    ).replace([np.inf, -np.inf], 0).fillna(0.0)

    df["volatility_10d"] = df["return"].rolling(window=10).std().fillna(0.0)

    # -------- 3) 최종 컬럼만 사용 --------
    df = df[["date", "close", "return", "log_return", "volatility_10d", "거래량"]]
    df = df.rename(columns={"거래량": "volume"})

    return df


def save_kospi():
    df = fetch_kospi_index(START_DATE, END_DATE)

    out_path = RAW_KOSPI_DIR / f"kospi_index_{START_DATE}_{END_DATE}.parquet"
    df.to_parquet(out_path, index=False)

    print("[KOSPI SAVED]")
    print(" →", out_path)
    print(" → rows:", len(df))

    return out_path


if __name__ == "__main__":
    save_kospi()
