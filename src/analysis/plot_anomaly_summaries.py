# src/analysis/generate_all_charts.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

from bertopic import BERTopic
from src.config import PROCESSED_DIR, TOPIC_DIR

FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    """진행 상황 출력 함수"""
    print(f"[INFO] {msg}")


def load_data():
    log("Loading anomaly_scores.parquet ...")
    df = pd.read_parquet(PROCESSED_DIR / "anomaly_scores.parquet")
    df = df.sort_values("date")
    log(f"Loaded dataframe with {len(df)} rows.")
    return df


# 1) KOSPI 가격 + 이상치
def plot_price_with_anomalies(df):
    log("Plotting: Price with anomalies ...")
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="KOSPI Close")
    tops = df.nlargest(20, "anomaly_score")
    plt.scatter(tops["date"], tops["close"], color="red", label="Top 20 Anomalies")
    plt.title("KOSPI Price with Top 20 Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_price_with_anomalies.png")
    plt.close()
    log("Saved: 01_price_with_anomalies.png")


# 2) 이상치 점수 추세
def plot_anomaly_scores(df):
    log("Plotting: Anomaly score trend ...")
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["anomaly_score"], label="Anomaly Score")
    tops = df.nlargest(20, "anomaly_score")
    plt.scatter(tops["date"], tops["anomaly_score"], color="red", label="Top 20")
    plt.title("Anomaly Score Trend")
    plt.xlabel("Date")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_anomaly_score_trend.png")
    plt.close()
    log("Saved: 02_anomaly_score_trend.png")


# 3) 토픽 분포 바차트
def plot_topic_distribution(df):
    log("Plotting: Topic distribution ...")
    topic_cols = [c for c in df.columns if c.startswith("topic_") and c != "topic_id"]

    if not topic_cols:
        log("[WARN] No topic_ columns found. Skipping topic chart.")
        return

    tops = df.nlargest(20, "anomaly_score")
    mean_topics = tops[topic_cols].mean().sort_values(ascending=False)

    try:
        log("Loading BERTopic model for topic labels ...")
        topic_model = BERTopic.load(TOPIC_DIR / "bertopic_model")

        topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
        topic_name_dict = dict(zip(topic_info["Topic"], topic_info["Name"]))

        def col_to_label(col_name: str) -> str:
            try:
                num = int(col_name.split("_")[1])
                label = topic_name_dict.get(num, f"Topic {num}")
                if len(label) > 40:
                    label = label[:37] + "..."
                return label
            except Exception:
                return col_name

        mean_topics.index = [col_to_label(c) for c in mean_topics.index]

    except Exception as e:
        log(f"[WARN] Failed to load BERTopic labels: {e}")

    plt.figure(figsize=(12, 6))
    mean_topics.plot(kind="bar")
    plt.title("Topic Distribution Among Top 20 Anomalies")
    plt.ylabel("Mean One-hot Ratio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_topic_distribution_labeled.png")
    plt.close()
    log("Saved: 03_topic_distribution_labeled.png")


# 4) 뉴스 요약용 Top20 테이블
def plot_summary_table(df):
    log("Plotting: Top 20 summary table ...")
    tops = df.nlargest(20, "anomaly_score")[["date", "anomaly_score"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    table = ax.table(
        cellText=tops.values,
        colLabels=tops.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.title("Top 20 Anomalies (Date & Score)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_top20_table.png")
    plt.close()
    log("Saved: 04_top20_table.png")


# 5) 변동성 그래프
def plot_volatility_analysis(df):
    log("Plotting: Volatility analysis ...")
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["volatility_10d"], label="10d Volatility")

    tops = df.nlargest(20, "anomaly_score")
    plt.scatter(tops["date"], tops["volatility_10d"], color="red", label="Top 20 Anomalies")

    plt.title("Volatility Change Around Anomalies")
    plt.xlabel("Date")
    plt.ylabel("10d Rolling Volatility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_volatility_analysis.png")
    plt.close()
    log("Saved: 05_volatility_analysis.png")


def generate_all():
    total_start = time.time()

    df = load_data()
    log("Generating charts...")

    plot_price_with_anomalies(df)
    plot_anomaly_scores(df)
    plot_topic_distribution(df)
    plot_summary_table(df)
    plot_volatility_analysis(df)

    log(f"All charts saved in: {FIG_DIR}")
    log(f"Total time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    generate_all()
