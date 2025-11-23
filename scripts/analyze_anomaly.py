# scripts/analyze_anomaly.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================
# 1. 유틸 함수들
# ============================

def load_anomaly_scores(path="../data/processed/anomaly_scores.csv") -> pd.DataFrame:
    """
    anomaly_scores.csv 로드
    (inference.py에서 저장한 파일)
    """
    df = pd.read_csv(path)

    # 날짜 정리
    df["날짜_dt"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜_dt").reset_index(drop=True)

    print("[STEP] Load anomaly_scores")
    print(" - shape:", df.shape)
    print(" - columns:", list(df.columns))
    print(" - 날짜 범위:", df["날짜_dt"].min().date(), "~", df["날짜_dt"].max().date())
    return df


def compute_thresholds(df: pd.DataFrame, col: str = "recon_loss"):
    """
    TOP N% + IQR 기준 threshold 계산
    """
    scores = df[col].values

    # TOP 5% 기준
    top5_threshold = np.percentile(scores, 95)

    # IQR 기준
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    iqr_threshold = q3 + 1.5 * iqr

    print("[INFO] Thresholds")
    print(f" - TOP 5% 기준 threshold : {top5_threshold:.4f}")
    print(f" - IQR 기반 threshold (Q3 + 1.5*IQR): {iqr_threshold:.4f}")
    print(f"   (Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f})")

    return top5_threshold, iqr_threshold


def attach_anomaly_flags(df: pd.DataFrame,
                         top5_threshold: float,
                         iqr_threshold: float,
                         col: str = "recon_loss") -> pd.DataFrame:
    """
    anomaly_scores에 이상치 플래그 컬럼 추가
    """
    df = df.copy()
    df["is_anomaly_top5"] = df[col] >= top5_threshold
    df["is_anomaly_iqr"] = df[col] >= iqr_threshold

    print("\n[INFO] Anomaly counts")
    print(" - TOP 5% 기준 이상치 수 :", df["is_anomaly_top5"].sum())
    print(" - IQR 기준 이상치 수           :", df["is_anomaly_iqr"].sum())

    return df


# ============================
# 2. 시각화 함수들
# ============================

def plot_price_with_anomaly_intensity(df: pd.DataFrame,
                                      use_flag: str = "is_anomaly_iqr",
                                      save_path="../figures/price_anomaly_iqr.png"):
    """
    종가 + 이상치 intensity (recon_loss) 함께 플롯
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 한국어 폰트 (윈도우 기준) - 경고 싫으면 설정
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(14, 6))

    # 종가 라인
    plt.plot(df["날짜_dt"], df["종가"], label="KOSPI 종가", alpha=0.6)

    # 이상치 intensity는 scatter로
    anomaly_df = df[df[use_flag]]
    plt.scatter(
        anomaly_df["날짜_dt"],
        anomaly_df["종가"],
        s=(anomaly_df["recon_loss"] ** 2),  # 재구성 오차^2 만큼 점 크기
        alpha=0.6,
        edgecolors="red",
        facecolors="none",
        label="Anomaly (IQR)"
    )

    plt.title("KOSPI 종가 및 이상치 intensity")
    plt.xlabel("날짜")
    plt.ylabel("종가")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[SAVE] ../figures/price_anomaly_iqr.png")


def plot_monthly_anomaly_heatmap(df: pd.DataFrame,
                                 use_flag: str = "is_anomaly_iqr",
                                 save_path="../figures/monthly_anomaly_heatmap_iqr.png"):
    """
    연-월별 이상치 발생 빈도 heatmap
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = df.copy()
    df["year"] = df["날짜_dt"].dt.year
    df["month"] = df["날짜_dt"].dt.month

    month_anom = (
        df[df[use_flag]]
        .groupby(["year", "month"])["날짜"]
        .count()
        .reset_index()
        .rename(columns={"날짜": "anomaly_cnt"})
    )

    pivot = month_anom.pivot(index="year", columns="month", values="anomaly_cnt")
    pivot = pivot.fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds")
    plt.title("월별 이상치 발생 횟수 (IQR 기준)")
    plt.xlabel("월")
    plt.ylabel("연도")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[SAVE] monthly anomaly heatmap → {save_path}")


# ============================
# 3. 날짜별 Top-5 뉴스 제목
# ============================

def print_top_titles_for_anomaly_days(
    df_anomaly: pd.DataFrame,
    news_raw_path="../data/raw/news_raw.csv",
    use_flag: str = "is_anomaly_iqr",
    top_k_days: int = 27,
    top_k_titles: int = 5,
):
    """
    anomaly 날짜별로 뉴스 제목 Top-5 출력 (사람이 읽기 좋게)
    - 여기서는 pubDate(기사 실제 시간)를 기준으로 '그 날' 뉴스만 사용
    - 같은 제목은 1번만 보여주도록 중복 제거
    """
    if not os.path.exists(news_raw_path):
        print(f"[WARN] news_raw 파일을 찾을 수 없습니다: {news_raw_path}")
        return

    news_raw = pd.read_csv(news_raw_path)

    # pubDate → 날짜
    if "pubDate" not in news_raw.columns:
        print("[WARN] news_raw에 pubDate 컬럼이 없습니다.")
        return

    news_raw["pubDate_dt"] = pd.to_datetime(news_raw["pubDate"]).dt.date

    # anomaly day들 정렬 (recon_loss 큰 순서)
    anom_days = (
        df_anomaly[df_anomaly[use_flag]]
        .sort_values("recon_loss", ascending=False)
        .head(top_k_days)
        .reset_index(drop=True)
    )

    print("\n[STEP] Top-5 뉴스 제목 per anomaly day\n")

    for _, row in anom_days.iterrows():
        day = row["날짜_dt"].date()
        loss_val = row["recon_loss"]
        print("-------------------------------")
        print(f"📌 날짜: {day} | recon_loss: {loss_val:.4f}")

        # 해당 anomaly 날짜의 기사만 사용 (pubDate 기준)
        day_news = news_raw[news_raw["pubDate_dt"] == day].copy()

        if day_news.empty:
            print(" - 해당 날짜의 뉴스 없음")
            continue

        # 제목 중복 제거
        day_news = day_news.drop_duplicates(subset="title")

        # pubDate 시간 기준 내림차순 정렬
        day_news["pubDate_ts"] = pd.to_datetime(day_news["pubDate"])
        day_news = day_news.sort_values("pubDate_ts", ascending=False)

        titles = day_news["title"].head(top_k_titles).tolist()

        print(" - Top 뉴스 제목:")
        for i, t in enumerate(titles, start=1):
            print(f"   {i}. {t}")

    print("\n[SUCCESS] Top-5 제목 출력 완료.")


# ============================
# 4. 메인 실행
# ============================

def main():
    print("\n========================")
    print("  Anomaly Analysis")
    print("========================\n")

    # 1) anomaly_scores 로드
    scores_path = "../data/processed/anomaly_scores.csv"
    df_scores = load_anomaly_scores(scores_path)

    # 2) threshold 계산 + flag 부여
    top5_thr, iqr_thr = compute_thresholds(df_scores, col="recon_loss")
    df_scores = attach_anomaly_flags(df_scores, top5_thr, iqr_thr, col="recon_loss")

    # 3) Top-10 anomaly day 출력 (뉴스텍스트 포함)
    print("\n[STEP] Top-10 anomaly days (기준: is_anomaly_iqr)\n")
    top10 = (
        df_scores[df_scores["is_anomaly_iqr"]]
        .sort_values("recon_loss", ascending=False)
        .head(10)
    )

    for _, row in top10.iterrows():
        print("-------------------------")
        print("날짜       :", row["날짜_dt"].date())
        print("종가       :", row["종가"])
        print("recon_loss :", round(row["recon_loss"], 4))
        txt = str(row.get("뉴스텍스트", ""))[:200]
        print("뉴스텍스트 :", txt, "\n")

    # 4) 시각화: 가격 + intensity
    print("[STEP] Plotting price + anomaly intensity...")
    plot_price_with_anomaly_intensity(df_scores, use_flag="is_anomaly_iqr")

    # 5) 시각화: 월별 heatmap
    print("\n[STEP] Plotting monthly anomaly heatmap...")
    plot_monthly_anomaly_heatmap(df_scores, use_flag="is_anomaly_iqr")

    # 6) anomaly 날짜별 Top-5 뉴스 제목
    print("\n[STEP] Top-5 뉴스 제목 per anomaly day")
    print_top_titles_for_anomaly_days(
        df_anomaly=df_scores,
        news_raw_path="../data/raw/news_raw.csv",
        use_flag="is_anomaly_iqr",
        top_k_days=27,       # IQR 이상치 전부
        top_k_titles=5,
    )

    print("\n[SUCCESS] Anomaly analysis complete.")


if __name__ == "__main__":
    main()
