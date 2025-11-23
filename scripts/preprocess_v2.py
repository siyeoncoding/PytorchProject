# scripts/preprocess_v2.py
import os
import re
import html
import numpy as np
import pandas as pd


# ----------------------------------------
# 유틸: 경로 자동 생성
# ----------------------------------------
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ----------------------------------------
# 유틸: 제목 전처리 (중복 집계용)
# ----------------------------------------
def clean_title(text: str) -> str:
    """HTML 태그 제거 + HTML entities 해제 + 공백 정리"""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)      # 태그 제거
    text = re.sub(r"\s+", " ", text)          # 공백 정리
    return text.strip()


# ========================================
# STEP 1 — KOSPI 전처리 (v2)
#   • 2024-10-01 ~ 2025-10-31 기간만 사용
#   • return, volatility 계산
# ========================================
def process_kospi_v2():
    raw_path = "../data/raw/kospi.csv"
    save_path = "../data/processed/kospi_processed_v2.csv"

    df = pd.read_csv(raw_path)

    # 날짜를 Timestamp로 통일
    df["날짜"] = pd.to_datetime(df["날짜"])

    # ---- 기간 필터 (2024-10-01 ~ 2025-10-31)
    start = pd.Timestamp("2024-10-01")
    end = pd.Timestamp("2025-10-31")
    df = df[(df["날짜"] >= start) & (df["날짜"] <= end)]

    df = df.sort_values("날짜").reset_index(drop=True)

    # 종가 숫자화
    df["종가"] = pd.to_numeric(df["종가"], errors="coerce")
    df["종가"] = df["종가"].fillna(method="ffill").fillna(0.0)

    # 수익률 / 변동성
    df["return"] = df["종가"].pct_change()
    df["return"] = df["return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["volatility"] = df["return"].rolling(window=20, min_periods=1).std()
    df["volatility"] = df["volatility"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print("[STEP 1] KOSPI processed (v2)")
    print(" - shape:", df.shape)
    print(" - date range:", df["날짜"].min(), "~", df["날짜"].max())

    ensure_dir(save_path)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {save_path}")
    return df


# ========================================
# STEP 2 — 뉴스 전처리 (v2)
#   • 2024-10-01 ~ 2025-10-31
#   • 중복 제거용 clean_title
#   • 날짜별 뉴스텍스트 / 뉴스개수 / 고유제목수
# ========================================
def process_news_v2():
    raw_path = "../data/raw/news_raw.csv"
    save_path = "../data/processed/news_daily_v2.csv"

    news = pd.read_csv(raw_path)

    # 날짜를 Timestamp로 통일
    news["date"] = pd.to_datetime(news["date"])

    # ---- 기간 필터
    start = pd.Timestamp("2024-10-01")
    end = pd.Timestamp("2025-10-31")
    news = news[(news["date"] >= start) & (news["date"] <= end)]

    # 결측 처리
    news["title"] = news["title"].fillna("")
    news["description"] = news["description"].fillna("")

    # 제목 클린 버전 (중복 집계용)
    news["title_clean"] = news["title"].apply(clean_title)

    # 텍스트 합치기 (제목 + 요약)
    news["뉴스텍스트"] = (news["title"] + " " + news["description"]).str.strip()

    # 날짜별 집계
    news_daily = (
        news.groupby("date")
        .agg(
            뉴스텍스트=("뉴스텍스트", lambda x: " ".join(x.astype(str))),
            뉴스개수=("title", "size"),
            고유제목수=("title_clean", pd.Series.nunique),
        )
        .reset_index()
    )

    # 날짜 컬럼 이름 맞추기
    news_daily = news_daily.rename(columns={"date": "날짜"})

    print("[STEP 2] NEWS processed (v2)")
    print(" - shape:", news_daily.shape)
    print(" - date range:", news_daily["날짜"].min(), "~", news_daily["날짜"].max())

    ensure_dir(save_path)
    news_daily.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {save_path}")
    return news_daily


# ========================================
# STEP 3 — Merge KOSPI + NEWS (v2)
#   • 뉴스 없는 날은 뉴스텍스트 "" / 뉴스개수 0 / 고유제목수 0
# ========================================
def merge_all_v2():
    kospi_path = "../data/processed/kospi_processed_v2.csv"
    news_path = "../data/processed/news_daily_v2.csv"
    save_path = "../data/processed/merged_kospi_news_v2.csv"

    # 날짜 컬럼 바로 파싱
    kospi = pd.read_csv(kospi_path, parse_dates=["날짜"])
    news = pd.read_csv(news_path, parse_dates=["날짜"])

    # LEFT JOIN (KOSPI 기준)
    merged = pd.merge(kospi, news, on="날짜", how="left")

    # 뉴스 없는 날 처리
    merged["뉴스텍스트"] = merged["뉴스텍스트"].fillna("")
    merged["뉴스개수"] = merged["뉴스개수"].fillna(0).astype(int)
    merged["고유제목수"] = merged["고유제목수"].fillna(0).astype(int)

    print("[STEP 3] Merge completed (v2)")
    print(" - shape:", merged.shape)
    print(" - date range:", merged["날짜"].min(), "~", merged["날짜"].max())
    print(" - 뉴스텍스트 NaN 수:", merged["뉴스텍스트"].isna().sum())

    ensure_dir(save_path)
    merged.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {save_path}")
    return merged


# ========================================
# 실행부
# ========================================
if __name__ == "__main__":
    print("============================")
    print("   Preprocess Pipeline v2   ")
    print("============================\n")

    kospi = process_kospi_v2()
    news = process_news_v2()
    merged = merge_all_v2()

    print("\n[SUCCESS] Full preprocess v2 complete.")
