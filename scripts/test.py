# plot_anomaly_v2.py  (또는 test.py 에 그대로 써도 됨)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# 1) 데이터 로드
# ============================================
# ✅ Windows 경로는 raw string (r"") 로!
DATA_PATH = Path(
    r"C:\MyProject\PytorchProject\data\processed\anomaly_scores_v2.csv"
)

def load_anomaly_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ✅ 날짜 컬럼 정리 (영어/한글 모두 대응)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    elif "날짜" in df.columns:   # ← 네 CSV는 이 케이스
        df["date"] = pd.to_datetime(df["날짜"])
    else:
        raise ValueError("CSV에 'date' / 'Date' / '날짜' 컬럼이 필요합니다.")

    # 날짜 기준 정렬
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ============================================
# 2) threshold 계산 함수 (TOP5%, IQR, MAD)
# ============================================
def compute_thresholds(df: pd.DataFrame, col: str = "recon_loss") -> dict:
    x = df[col].values

    # TOP 5% 기준
    top5 = np.quantile(x, 0.95)

    # IQR 기반
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    iqr_thr = q3 + 1.5 * iqr

    # MAD 기반
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    mad_thr = median + 3.0 * mad

    print("[INFO] Thresholds")
    print(f" - TOP 5%: {top5:.4f}")
    print(f" - IQR:    {iqr_thr:.4f} (Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f})")
    print(f" - MAD:    {mad_thr:.4f} (median={median:.4f}, MAD={mad:.4f})")

    return {
        "top5": top5,
        "iqr": iqr_thr,
        "mad": mad_thr,
        "q1": q1,
        "q3": q3,
        "iqr_raw": iqr,
        "median": median,
        "mad_raw": mad,
    }


# ============================================
# 3) 이상치 플래그 생성
# ============================================
def add_anomaly_flags(df: pd.DataFrame, col: str, th: dict) -> pd.DataFrame:
    # 새 기준으로 플래그 3개 추가 (기존 is_anomaly_* 와는 별개)
    df["is_anom_top5"] = df[col] >= th["top5"]
    df["is_anom_iqr"] = df[col] >= th["iqr"]
    df["is_anom_mad"] = df[col] >= th["mad"]

    print(f"[INFO] is_anom_top5: {df['is_anom_top5'].sum()} 개 / {len(df)}")
    print(f"[INFO] is_anom_iqr:  {df['is_anom_iqr'].sum()} 개 / {len(df)}")
    print(f"[INFO] is_anom_mad:  {df['is_anom_mad'].sum()} 개 / {len(df)}")

    return df


# ============================================
# 4) 시계열 그래프 그리기
#  - 전체 recon_loss 추세 + 이상치 점 찍기
# ============================================
def plot_time_series_with_anomalies(
    df: pd.DataFrame,
    score_col: str = "recon_loss",
    flag_col: str = "is_anom_top5",   # 기본: 우리가 새로 만든 top5 기준
):
    plt.figure(figsize=(14, 6))

    # 전체 reconstruction loss 추세
    plt.plot(df["date"], df[score_col],
             label="Reconstruction Loss",
             linewidth=1.5)

    # 이상치 점
    anom = df[df[flag_col]]
    plt.scatter(anom["date"], anom[score_col],
                s=40,
                label=f"Anomaly ({flag_col})",
                alpha=0.9)

    plt.xlabel("Date")
    plt.ylabel("Reconstruction Loss")
    plt.title("1년치 Reconstruction Loss & Anomaly Points")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================
# 5) 월별 이상치 개수 막대 그래프
# ============================================
def plot_monthly_anomaly_counts(df: pd.DataFrame,
                                flag_col: str = "is_anom_top5"):
    # 날짜를 index로 두고 월별 resample
    tmp = df.set_index("date")[flag_col].resample("M").sum()

    plt.figure(figsize=(10, 4))
    tmp.plot(kind="bar")
    plt.title(f"월별 이상치 개수 ({flag_col})")
    plt.ylabel("Anomaly Count")
    plt.xlabel("Month")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================
# 메인 실행부
# ============================================
if __name__ == "__main__":
    df = load_anomaly_scores(DATA_PATH)

    # 필요하면 1년 기간 필터링
    # start = pd.Timestamp("2024-10-02")
    # end   = pd.Timestamp("2025-10-31")
    # df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    # threshold 계산 + 이상치 플래그 생성
    thresholds = compute_thresholds(df, col="recon_loss")
    df = add_anomaly_flags(df, col="recon_loss", th=thresholds)

    # (1) 시계열 + 이상치 점
    plot_time_series_with_anomalies(
        df,
        score_col="recon_loss",
        flag_col="is_anom_top5",   # 원하면 is_anom_iqr / is_anom_mad 로 바꿔도 됨
    )

    # (2) 월별 이상치 카운트
    plot_monthly_anomaly_counts(
        df,
        flag_col="is_anom_top5",
    )
