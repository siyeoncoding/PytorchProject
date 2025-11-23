# src/analysis/generate_all_charts.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from bertopic import BERTopic  # ⭐ 토픽 라벨 불러오기
from src.config import PROCESSED_DIR, TOPIC_DIR  # INTERIM_DIR는 지금 안 써서 제거해도 됨

FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """
    anomaly_scores.parquet 하나만 사용.
    (여기에 이미 topic_* 컬럼이 들어있으니까 굳이 다시 merge 안 함)
    """
    df = pd.read_parquet(PROCESSED_DIR / "anomaly_scores.parquet")
    df = df.sort_values("date")
    print("[INFO] columns:", df.columns.tolist())
    return df


# 1) KOSPI 가격 + 이상치
def plot_price_with_anomalies(df):
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


# 2) 이상치 점수 추세
def plot_anomaly_scores(df):
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


# 3) 토픽 분포 바차트 (⭐ 토픽 라벨 자동 적용)
def plot_topic_distribution(df):
    # topic_ 로 시작하지만 topic_id 는 제외
    topic_cols = [
        c for c in df.columns
        if c.startswith("topic_") and c != "topic_id"
    ]

    if not topic_cols:
        print("[WARN] topic_ 컬럼이 없어 토픽 분포 그래프는 생략합니다.")
        return

    tops = df.nlargest(20, "anomaly_score")
    mean_topics = tops[topic_cols].mean().sort_values(ascending=False)

    # ── BERTopic 모델에서 토픽 라벨 불러오기 ──
    try:
        topic_model = BERTopic.load(TOPIC_DIR / "bertopic_model")
        topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
        topic_name_dict = dict(zip(topic_info["Topic"], topic_info["Name"]))

        def col_to_label(col_name: str) -> str:
            # 'topic_3' -> 3 -> '금리 / 연준 / 인상 ...'
            try:
                num = int(col_name.split("_")[1])
                label = topic_name_dict.get(num, f"Topic {num}")
                # 너무 길면 앞부분만 자르기 (PPT 보기 좋게)
                if len(label) > 40:
                    label = label[:37] + "..."
                return label
            except Exception:
                return col_name

        labeled_index = [col_to_label(c) for c in mean_topics.index]
        mean_topics.index = labeled_index

    except Exception as e:
        print("[WARN] BERTopic 라벨 로딩 실패, 숫자 토픽 이름 그대로 사용합니다.")
        print("      →", e)

    # ── 바 차트 그리기 ──
    plt.figure(figsize=(12, 6))
    mean_topics.plot(kind="bar")
    plt.title("Topic Distribution Among Top 20 Anomalies")
    plt.ylabel("Mean One-hot Ratio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_topic_distribution_labeled.png")
    plt.close()


# 4) 뉴스 요약용 Top20 테이블 (지금은 날짜+점수만)
def plot_summary_table(df):
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


# 5) 변동성 변화 분석 그래프
def plot_volatility_analysis(df):
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


def generate_all():
    df = load_data()
    print("[INFO] Data Loaded")
    print("Generating charts...")

    plot_price_with_anomalies(df)
    plot_anomaly_scores(df)
    plot_topic_distribution(df)   # ⭐ 라벨 적용된 토픽 그래프
    plot_summary_table(df)
    plot_volatility_analysis(df)

    print("[DONE] Charts saved to:", FIG_DIR)


if __name__ == "__main__":
    generate_all()
