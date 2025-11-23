# # scripts/check_merged_rows.py
# import pandas as pd
#
# path = "../data/processed/merged_kospi_news.csv"
#
# df = pd.read_csv(path)
#
# print("▶ shape:", df.shape)          # (row 수, column 수)
# print("▶ row 수:", len(df))
# print("▶ 날짜 unique 개수:", df["날짜"].nunique())
# print("▶ 날짜 범위:", df["날짜"].min(), "~", df["날짜"].max())
# scripts/check_merge_debug.py

import pandas as pd

kospi_path = "../data/processed/kospi_processed.csv"
news_path = "../data/raw/news_raw.csv"
merged_path = "../data/processed/merged_kospi_news.csv"

def inspect(path, date_col):
    df = pd.read_csv(path)
    print(f"\n=== {path} ===")
    print("shape:", df.shape)
    print("date col:", date_col)
    print("unique 날짜 개수:", df[date_col].nunique())
    print("날짜 범위:", df[date_col].min(), "~", df[date_col].max())
    return df

if __name__ == "__main__":
    kospi = inspect(kospi_path, "날짜")
    # news_raw는 date라는 컬럼 이름이라고 가정
    news = inspect(news_path, "date")
    merged = inspect(merged_path, "날짜")
