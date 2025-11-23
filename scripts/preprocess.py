# scripts/preprocess.py
import os
import numpy as np
import pandas as pd


# -----------------------------
# STEP 1: KOSPI 가공
# -----------------------------
def process_kospi():
    raw_path = "../data/raw/kospi.csv"
    save_path = "../data/processed/kospi_processed.csv"

    df = pd.read_csv(raw_path)

    # 날짜 정리
    df["날짜"] = pd.to_datetime(df["날짜"]).dt.date
    df = df.sort_values("날짜").reset_index(drop=True)

    # 종가 숫자 변환
    df["종가"] = pd.to_numeric(df["종가"], errors="coerce")

    # ===== 수익률 / 변동성 계산 =====
    df["return"] = df["종가"].pct_change()
    df["volatility"] = df["return"].rolling(window=20, min_periods=1).std()

    # === 🚨 NaN / Inf 값 완전 제거 ===
    df["return"] = df["return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["volatility"] = df["volatility"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["종가"] = df["종가"].fillna(method="ffill").fillna(0.0)

    print("[STEP 1] process KOSPI")
    print(f"[SAVE] {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return df


# -----------------------------
# STEP 2: NEWS 가공 (1일 1문장으로 집계)
# -----------------------------
def process_news():
    raw_path = "../data/raw/news_raw.csv"
    save_path = "../data/processed/news_daily.csv"

    news = pd.read_csv(raw_path)

    # 날짜 정리
    news["date"] = pd.to_datetime(news["date"]).dt.date

    # 뉴스텍스트 생성: 제목 + 요약
    news["뉴스텍스트"] = (
        news["title"].fillna("") + " " + news["description"].fillna("")
    )

    # 날짜별로 전체 텍스트 합치기
    news_daily = (
        news.groupby("date")["뉴스텍스트"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    print("[STEP 2] process NEWS")
    print(" - news_raw shape:", news.shape)
    print(" - news_daily shape:", news_daily.shape)
    print(" - date range:", news_daily["date"].min(), "~", news_daily["date"].max())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    news_daily.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {save_path}")

    return news_daily


# -----------------------------
# STEP 3: MERGE
# -----------------------------
def merge_all():
    kospi_path = "../data/processed/kospi_processed.csv"
    news_daily_path = "../data/processed/news_daily.csv"
    save_path = "../data/processed/merged_kospi_news.csv"

    kospi = pd.read_csv(kospi_path)
    news_daily = pd.read_csv(news_daily_path)

    # 날짜 타입 맞추기
    kospi["날짜"] = pd.to_datetime(kospi["날짜"]).dt.date

    if "date" in news_daily.columns:
        news_daily["날짜"] = pd.to_datetime(news_daily["date"]).dt.date
        news_daily = news_daily.drop(columns=["date"])

    # 코스피 기준 left join
    merged = pd.merge(kospi, news_daily, on="날짜", how="left")

    # ===== 🚨 뉴스텍스트 NaN은 빈 문자열로 처리 =====
    merged["뉴스텍스트"] = merged["뉴스텍스트"].fillna("")

    # ===== 🚨 수치형 NaN 다시 한 번 체크 =====
    for col in ["종가", "return", "volatility"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print("\n[STEP 3] merge ALL")
    print("\n[INFO] MERGED RESULT")
    print("shape:", merged.shape)
    print("date range:", merged["날짜"].min(), "~", merged["날짜"].max())
    print("news NA:", merged["뉴스텍스트"].isna().sum())
    print("return NaN:", merged["return"].isna().sum())
    print("volatility NaN:", merged["volatility"].isna().sum())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {save_path}")

    return merged


# -----------------------------
# RUN ALL
# -----------------------------
if __name__ == "__main__":
    kospi = process_kospi()
    news_daily = process_news()
    merged = merge_all()
    print("\n[SUCCESS] Full preprocess pipeline complete.")
