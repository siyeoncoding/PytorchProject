# src/data/merge_modalities.py
import pandas as pd

from src.config import RAW_KOSPI_DIR, INTERIM_DIR


KOSPI_PATH = list(RAW_KOSPI_DIR.glob("kospi_index_*.parquet"))[0]
NEWS_PATH = INTERIM_DIR / "daily_news_merged.parquet"


def merge_news_kospi():
    kospi = pd.read_parquet(KOSPI_PATH)
    news = pd.read_parquet(NEWS_PATH)

    merged = pd.merge(kospi, news, how="left", on="date")

    merged["merged_text"] = merged["merged_text"].fillna("")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    out_path = INTERIM_DIR / "merged_timeseries.parquet"
    merged.to_parquet(out_path, index=False)

    print("[MERGED TIMESERIES SAVED]")
    print(" →", out_path)
    print(" → rows:", len(merged))

    return out_path


if __name__ == "__main__":
    merge_news_kospi()
