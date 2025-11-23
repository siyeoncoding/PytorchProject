# src/data/preprocess_news.py
import re
import html
import pandas as pd
from bs4 import BeautifulSoup

from src.config import RAW_NEWS_DIR, INTERIM_DIR


MERGED_NEWS_FILE = RAW_NEWS_DIR / "naver_news_merged.parquet"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_daily_news():
    if not MERGED_NEWS_FILE.exists():
        raise FileNotFoundError(MERGED_NEWS_FILE)

    df = pd.read_parquet(MERGED_NEWS_FILE)

    df["title_clean"] = df["title"].apply(clean_text)
    df["desc_clean"] = df["description"].apply(clean_text)

    df["date"] = pd.to_datetime(df["pub_datetime"]).dt.date
    df["text"] = df["title_clean"] + " " + df["desc_clean"]

    grouped = (
        df.groupby("date")["text"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"text": "merged_text"})
    )

    grouped["news_count"] = (
        df.groupby("date")["text"].size().values
    )

    out_path = INTERIM_DIR / "daily_news_merged.parquet"
    grouped.to_parquet(out_path, index=False)

    print("[NEWS MERGED SAVED]")
    print(" →", out_path)
    print(" → rows:", len(grouped))

    return out_path


if __name__ == "__main__":
    build_daily_news()
