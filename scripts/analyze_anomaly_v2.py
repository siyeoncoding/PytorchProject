# scripts/analyze_anomaly_v2.py
import os
import re
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#  공통 유틸: 제목/텍스트 클리닝 (중복 집계용)
# ============================================================
def clean_title(text: str) -> str:
    """
    HTML 태그 제거 + HTML entities 해제 + 공백 정리
    (preprocess_v2.py 의 clean_title 과 동일 컨셉)
    """
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)   # 태그 제거
    text = re.sub(r"\s+", " ", text)       # 공백 정리
    return text.strip()


# ============================================================
#  Threshold 재계산 (옵션: 분석용)
# ============================================================
def compute_thresholds(scores: np.ndarray, top_percent: float = 5.0, mad_k: float = 3.0):
    """
    recon_loss 배열(scores)을 받아서
    - TOP N% (상위 퍼센타일)
    - IQR (Q3 + 1.5*IQR)
    - MAD (median + k * MAD)
    기준 threshold를 계산해서 리턴
    """
    scores = np.asarray(scores, dtype=float)

    # TOP N% 기준
    top_thr = np.quantile(scores, 1.0 - top_percent / 100.0)

    # IQR 기준 (Q3 + 1.5*IQR)
    q1 = np.quantile(scores, 0.25)
    q3 = np.quantile(scores, 0.75)
    iqr = q3 - q1
    iqr_thr = q3 + 1.5 * iqr

    # MAD 기준 (median + k * MAD)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mad_thr = median + mad_k * mad

    thresholds = {
        "topN": top_thr,
        "iqr": iqr_thr,
        "mad": mad_thr,
        "q1": q1,
        "q3": q3,
        "iqr_raw": iqr,
        "median": median,
        "mad_raw": mad,
    }
    return thresholds


# ============================================================
#  Plot 1: 가격 + anomaly intensity (recon_loss)
# ============================================================
def plot_price_anomaly_intensity(
    df: pd.DataFrame,
    anomaly_flag_col: str = "is_anomaly_iqr",
    save_path: str = "../figures/price_anomaly_intensity_v2.png",
):
    """
    종가 + recon_loss + 이상치 구간을 한 번에 보는 플롯
    - 왼쪽 y축: 종가
    - 오른쪽 y축: recon_loss
    - 이상치 구간: 빨간 점(or marker)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_plot = df.copy()
    df_plot["날짜_dt"] = pd.to_datetime(df_plot["날짜"])

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 가격 (왼쪽 y축)
    ax1.plot(df_plot["날짜_dt"], df_plot["종가"], label="종가", linewidth=1.5)
    ax1.set_xlabel("날짜")
    ax1.set_ylabel("종가")

    # recon_loss (오른쪽 y축)
    ax2 = ax1.twinx()
    ax2.plot(df_plot["날짜_dt"], df_plot["recon_loss"], label="recon_loss", alpha=0.5, linewidth=1.0)
    ax2.set_ylabel("reconstruction loss")

    # 이상치 포인트 강조
    if anomaly_flag_col in df_plot.columns:
        anomalies = df_plot[df_plot[anomaly_flag_col]]
        ax2.scatter(
            anomalies["날짜_dt"],
            anomalies["recon_loss"],
            marker="o",
            s=50,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
            label="Anomaly"
        )

    # 범례
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title(f"Price & Anomaly Intensity (기준: {anomaly_flag_col})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[SAVE] price + anomaly intensity → {save_path}")


# ============================================================
#  Plot 2: 연/월 anomaly heatmap
# ============================================================
def plot_monthly_anomaly_heatmap(
    df: pd.DataFrame,
    anomaly_flag_col: str = "is_anomaly_iqr",
    save_path: str = "../figures/monthly_anomaly_heatmap_v2.png",
):
    """
    연-월 단위로 이상치 일수 카운트를 heatmap 으로 시각화
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_hm = df.copy()
    df_hm["날짜_dt"] = pd.to_datetime(df_hm["날짜"])
    df_hm["year"] = df_hm["날짜_dt"].dt.year
    df_hm["month"] = df_hm["날짜_dt"].dt.month

    if anomaly_flag_col in df_hm.columns:
        df_hm["is_anomaly"] = df_hm[anomaly_flag_col].astype(int)
    else:
        df_hm["is_anomaly"] = 0

    monthly = (
        df_hm.groupby(["year", "month"])["is_anomaly"]
        .sum()
        .reset_index()
        .pivot(index="year", columns="month", values="is_anomaly")
        .fillna(0)
        .astype(int)
    )

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        monthly,
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar_kws={"label": "Anomaly Days"},
    )
    plt.title(f"Monthly Anomaly Heatmap (기준: {anomaly_flag_col})")
    plt.xlabel("월")
    plt.ylabel("연도")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[SAVE] monthly anomaly heatmap → {save_path}")


# ============================================================
#  뉴스 raw 에서 날짜별 Top-5 제목(중복 카운트) 계산
# ============================================================
def build_daily_top_titles(
    news_raw_path: str = "../data/raw/news_raw.csv",
    top_k: int = 5,
):
    """
    news_raw.csv 에서 날짜별로
      - 제목 clean
      - 동일 제목 중복 카운트
      - 날짜별 Top-K (제목, count) list 생성
    return: dict[date(파이썬 date)] = [(title, count), ...]
    """
    if not os.path.exists(news_raw_path):
        print(f"[WARN] news_raw file not found: {news_raw_path}")
        return {}

    news = pd.read_csv(news_raw_path, encoding="utf-8-sig")

    if "date" not in news.columns or "title" not in news.columns:
        print("[WARN] news_raw.csv 에 'date' 또는 'title' 컬럼이 없습니다.")
        return {}

    news["date_dt"] = pd.to_datetime(news["date"]).dt.date
    news["title"] = news["title"].fillna("")
    news["title_clean"] = news["title"].apply(clean_title)
    news = news[news["title_clean"] != ""]  # 빈 제목 제거

    daily_title_dict = {}

    for d, group in news.groupby("date_dt"):
        # clean_title 기준으로 count
        cnt = group.groupby("title_clean").size().sort_values(ascending=False)

        top_list = []
        for clean_t, c in cnt.head(top_k).items():
            # 대표 원문 제목 하나 가져오기
            raw_example = group[group["title_clean"] == clean_t]["title"].iloc[0]
            top_list.append((raw_example, int(c)))

        daily_title_dict[d] = top_list

    print(f"[STEP] Built daily top-{top_k} titles from news_raw (days: {len(daily_title_dict)})")
    return daily_title_dict


# ============================================================
#  메인 분석 함수
# ============================================================
def analyze_anomaly_v2(
    anomaly_csv_path: str = "../data/processed/anomaly_scores_v2.csv",
    news_raw_path: str = "../data/raw/news_raw.csv",
    anomaly_flag_col: str = "is_anomaly_iqr",  # 기본: IQR 기준
    top_percent: float = 5.0,
    mad_k: float = 3.0,
    top_n_days: int = 20,
):
    print("========================")
    print("  Anomaly Analysis (v2) ")
    print("========================\n")

    # -----------------------------
    # 1) anomaly_scores_v2.csv 로드
    # -----------------------------
    if not os.path.exists(anomaly_csv_path):
        raise FileNotFoundError(f"anomaly_scores_v2.csv not found: {anomaly_csv_path}")

    df = pd.read_csv(anomaly_csv_path, encoding="utf-8-sig")

    if "날짜" not in df.columns:
        raise KeyError("'날짜' 컬럼이 anomaly_scores_v2.csv 에 없습니다.")

    df["날짜_dt"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜_dt").reset_index(drop=True)

    print("[STEP] Load anomaly_scores_v2")
    print(f" - shape: {df.shape}")
    print(f" - date range: {df['날짜_dt'].min().date()} ~ {df['날짜_dt'].max().date()}")

    if "recon_loss" not in df.columns:
        raise KeyError("'recon_loss' 컬럼이 anomaly_scores_v2.csv 에 없습니다.")

    # -----------------------------
    # 2) Threshold 다시 계산 (옵션)
    # -----------------------------
    thresholds = compute_thresholds(df["recon_loss"].values, top_percent=top_percent, mad_k=mad_k)

    print("\n[INFO] Thresholds (recomputed from recon_loss)")
    print(f" - TOP {top_percent:.1f}% 기준 threshold : {thresholds['topN']:.4f}")
    print(
        f" - IQR 기반 threshold (Q3 + 1.5*IQR): {thresholds['iqr']:.4f} "
        f"(Q1={thresholds['q1']:.4f}, Q3={thresholds['q3']:.4f}, IQR={thresholds['iqr_raw']:.4f})"
    )
    print(
        f" - MAD 기반 threshold (median + {mad_k:.1f} * MAD): {thresholds['mad']:.4f} "
        f"(median={thresholds['median']:.4f}, MAD={thresholds['mad_raw']:.4f})"
    )

    # -----------------------------
    # 3) anomaly flag 요약
    # -----------------------------
    for col in ["is_anomaly_topN", "is_anomaly_iqr", "is_anomaly_mad"]:
        if col in df.columns:
            print(f" - {col}: {df[col].sum()} 개 / {len(df)}")

    if anomaly_flag_col not in df.columns:
        raise KeyError(f"'{anomaly_flag_col}' 컬럼이 anomaly_scores_v2.csv 에 없습니다.")

    # -----------------------------
    # 4) 뉴스개수 / 고유제목수 전일 대비 변화
    # -----------------------------
    if "뉴스개수" in df.columns:
        df["뉴스개수_diff"] = df["뉴스개수"].diff().fillna(0).astype(int)
    else:
        df["뉴스개수"] = 0
        df["뉴스개수_diff"] = 0

    if "고유제목수" in df.columns:
        df["고유제목수_diff"] = df["고유제목수"].diff().fillna(0).astype(int)
    else:
        df["고유제목수"] = 0
        df["고유제목수_diff"] = 0

    # -----------------------------
    # 5) 이상치 일자 정렬 (recon_loss 기준)
    # -----------------------------
    anomalies = df[df[anomaly_flag_col]].copy()
    anomalies = anomalies.sort_values("recon_loss", ascending=False).reset_index(drop=True)

    print(f"\n[STEP] Top-{top_n_days} anomaly days (기준: {anomaly_flag_col}, 정렬: recon_loss 내림차순)\n")

    # -----------------------------
    # 6) 날짜별 Top-5 뉴스 제목 사전 구성
    # -----------------------------
    daily_title_dict = build_daily_top_titles(news_raw_path=news_raw_path, top_k=5)

    # -----------------------------
    # 7) 상위 N개 이상치 날짜 상세 출력
    # -----------------------------
    for i in range(min(top_n_days, len(anomalies))):
        row = anomalies.iloc[i]
        d = row["날짜_dt"].date()
        close_price = row.get("종가", np.nan)
        recon_loss = row["recon_loss"]

        news_count = row.get("뉴스개수", 0)
        uniq_count = row.get("고유제목수", 0)
        news_diff = row.get("뉴스개수_diff", 0)
        uniq_diff = row.get("고유제목수_diff", 0)

        print("-------------------------------")
        print(f"📌 날짜: {d} | 종가: {close_price} | recon_loss: {recon_loss:.4f}")
        print(
            f" - 뉴스개수: {news_count} (전일 대비 {news_diff:+d}) "
            f"/ 고유제목수: {uniq_count} (전일 대비 {uniq_diff:+d})"
        )

        # 해당 날짜의 Top-5 뉴스 제목
        titles = daily_title_dict.get(d, [])
        if not titles:
            print(" - 해당 날짜의 뉴스 없음 (news_raw 기준)")
        else:
            print(" - Top 뉴스 제목:")
            for j, (title, cnt) in enumerate(titles, start=1):
                print(f"   {j}. {title} (count={cnt})")

    print("\n[SUCCESS] Top-5 뉴스 제목 + 뉴스 개수 변화 출력 완료.\n")

    # -----------------------------
    # 8) 시각화 (intensity plot + heatmap)
    # -----------------------------
    print("[STEP] Plotting price + anomaly intensity...")
    plot_price_anomaly_intensity(
        df=df,
        anomaly_flag_col=anomaly_flag_col,
        save_path="../figures/price_anomaly_intensity_v2.png",
    )

    print("[STEP] Plotting monthly anomaly heatmap...")
    plot_monthly_anomaly_heatmap(
        df=df,
        anomaly_flag_col=anomaly_flag_col,
        save_path="../figures/monthly_anomaly_heatmap_v2.png",
    )

    print("\n[SUCCESS] Anomaly analysis v2 complete.\n")


# ============================================================
#  실행부
# ============================================================
if __name__ == "__main__":
    analyze_anomaly_v2(
        anomaly_csv_path="../data/processed/anomaly_scores_v2.csv",
        news_raw_path="../data/raw/news_raw.csv",
        anomaly_flag_col="is_anomaly_iqr",  # 필요하면 is_anomaly_topN, is_anomaly_mad 로 바꿔도 됨
        top_percent=5.0,
        mad_k=3.0,
        top_n_days=20,
    )
