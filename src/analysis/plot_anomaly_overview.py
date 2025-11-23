# src/analysis/plot_anomaly_overview.py

import textwrap
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

from src.config import PROCESSED_DIR
import matplotlib as mpl

# Windows 한글 폰트 설정
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False


FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[INFO] {msg}")


def load_anomaly_data():
    path = PROCESSED_DIR / "anomaly_scores.parquet"
    log(f"Loading anomaly scores from: {path}")
    df = pd.read_parquet(path)
    df = df.sort_values("date")
    log(f"Loaded {len(df)} rows")
    return df


def get_top20_with_summary(df, model_name="gogamza/kobart-summarization"):
    """상위 20개 이상치 날짜 + 뉴스 요약 생성 후 DataFrame 반환"""
    # 1) Top20 선택
    top = df.nlargest(20, "anomaly_score").copy()

    # 2) 요약 모델 로드
    log(f"Loading summarization model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)

    summaries = []
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        date = row["date"]
        score = row["anomaly_score"]
        close = row["close"]
        text = str(row.get("merged_text", ""))[:2048]

        log(f"[{i}/20] summarizing {date} (score={score:.6f}, close={close:.2f})")

        if not text.strip():
            summaries.append("뉴스 텍스트 없음")
            continue

        summ = summarizer(text)[0]["summary_text"]
        summaries.append(summ)

    top["summary"] = summaries
    return top


def plot_price_with_anomalies(df, top):
    """KOSPI 가격 + 이상치 위치 표시 그래프"""
    log("Plotting price with anomalies...")

    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="KOSPI 종가")
    plt.scatter(top["date"], top["close"], color="red", label="이상치 TOP 20")

    plt.title("KOSPI 가격과 이상치(Top 20)")
    plt.xlabel("날짜")
    plt.ylabel("종가")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "A1_price_with_anomalies_top20.png"
    plt.savefig(out_path)
    plt.close()
    log(f"Saved figure: {out_path}")


def plot_top20_summary_table(top):
    """Top20 날짜 + 점수 + 요약을 표 형태로 이미지로 저장"""
    log("Plotting Top20 summary table...")

    # 표에 들어갈 컬럼 정리
    df_tbl = top[["date", "close", "anomaly_score", "summary"]].copy()

    # 요약은 너무 길면 잘라서 한 줄로
    def shorten(s):
        return textwrap.shorten(s, width=80, placeholder="...")

    df_tbl["summary_short"] = df_tbl["summary"].apply(shorten)

    # 표시용 컬럼 순서/이름
    display_cols = ["date", "close", "anomaly_score", "summary_short"]
    col_labels = ["날짜", "종가", "Anomaly Score", "뉴스 요약(요약본)"]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")

    table = ax.table(
        cellText=df_tbl[display_cols].values,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    plt.title("이상치 Top 20 날짜별 뉴스 요약", pad=20)
    plt.tight_layout()

    out_path = FIG_DIR / "A2_top20_anomaly_summaries.png"
    plt.savefig(out_path)
    plt.close()
    log(f"Saved figure: {out_path}")


def main():
    df = load_anomaly_data()
    top = get_top20_with_summary(df)

    # 그래프 1: 가격 + 이상치
    plot_price_with_anomalies(df, top)

    # 그래프 2: Top20 요약 표
    plot_top20_summary_table(top)

    log("Done. Two figures ready for PPT:")

    print(" - A1_price_with_anomalies_top20.png")
    print(" - A2_top20_anomaly_summaries.png")


if __name__ == "__main__":
    main()
