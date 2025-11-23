# src/analysis/generate_all_charts.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from bertopic import BERTopic
from src.config import PROCESSED_DIR, TOPIC_DIR

# ============================
# Matplotlib 한글 폰트 설정
# ============================
import matplotlib as mpl

# 윈도우 기준: 맑은 고딕 사용
mpl.rcParams["font.family"] = "Malgun Gothic"
# 마이너스 깨짐 방지
mpl.rcParams["axes.unicode_minus"] = False

# 그림 저장 폴더
FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================
# 데이터 로드
# ============================
def load_data():
    """
    anomaly_scores.parquet 하나만 사용.
    (여기에 이미 close, volatility_10d, topic_* 등이 들어있다고 가정)
    """
    df = pd.read_parquet(PROCESSED_DIR / "anomaly_scores.parquet")
    df = df.sort_values("date")
    print("[INFO] columns:", df.columns.tolist())
    return df


# ============================
# 1) KOSPI 가격 + 이상치 점 표시
# ============================
def plot_price_with_anomalies(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="KOSPI 종가")
    tops = df.nlargest(20, "anomaly_score")
    plt.scatter(tops["date"], tops["close"], color="red", label="이상치 TOP 20")

    plt.title("KOSPI 가격과 이상치(Top 20)")
    plt.xlabel("날짜")
    plt.ylabel("종가")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_price_with_anomalies.png")
    plt.close()


# ============================
# 2) 이상치 점수 추세
# ============================
def plot_anomaly_scores(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["anomaly_score"], label="재구성 오차(Anomaly Score)")
    tops = df.nlargest(20, "anomaly_score")
    plt.scatter(tops["date"], tops["anomaly_score"], color="red", label="이상치 TOP 20")

    plt.title("Anomaly Score 추세")
    plt.xlabel("날짜")
    plt.ylabel("재구성 오차")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_anomaly_score_trend.png")
    plt.close()


# ============================
# 3) 토픽 분포 바 차트 (한글 라벨 적용)
# ============================
def plot_topic_distribution(df):
    # topic_id 컬럼이 없다면 스킵
    if "topic_id" not in df.columns:
        print("[WARN] topic_id 없어서 토픽 분석 스킵")
        return

    # top 20 anomaly
    tops = df.nlargest(20, "anomaly_score")

    # 원핫 컬럼 자동 탐색
    topic_cols = [c for c in df.columns if c.startswith("topic_") and c != "topic_id"]
    if not topic_cols:
        print("[WARN] topic_xx 컬럼 없음")
        return

    # 평균 one-hot 비율
    mean_topics = tops[topic_cols].mean().sort_values(ascending=False)

    # ───────────────────────────────
    #   BERTopic에서 "키워드"만 추출 (기사 문장 제거)
    # ───────────────────────────────
    try:
        topic_model = BERTopic.load(TOPIC_DIR / "bertopic_model")
        raw_topics = topic_model.get_topics()

        # 한글용 불용어/필터 정의
        stopwords = {
            "기사", "기사는", "기사입니다", "사진", "연합뉴스", "뉴스",
            "영상", "제공", "서울", "한국", "코스피지수", "코스닥지수"
        }

        def is_noise(word: str) -> bool:
            # 숫자+년/월, 순수 숫자, 불용어 등 제거
            if word in stopwords:
                return True
            if word.endswith("년") or word.endswith("월"):
                return True
            if word.isdigit():
                return True
            if len(word) <= 1:  # 한 글자 짜리
                return True
            return False

        topic_keywords = {}
        for tid, word_list in raw_topics.items():
            if tid == -1:
                continue  # outlier topic

            kws = []
            for w, _ in word_list:
                if is_noise(w):
                    continue
                if w not in kws:
                    kws.append(w)
                if len(kws) == 4:  # 최대 4개만
                    break

            if kws:
                label = " / ".join(kws)
            else:
                label = f"Topic {tid}"

            if len(label) > 30:
                label = label[:27] + "..."
            topic_keywords[tid] = label

        # 'topic_3' → 3 → "금리 / 연준 / 인상"
        clean_labels = []
        for col in mean_topics.index:
            try:
                tnum = int(col.split("_")[1])
                clean_labels.append(topic_keywords.get(tnum, f"Topic {tnum}"))
            except Exception:
                clean_labels.append(col)

        mean_topics.index = clean_labels

    except Exception as e:
        print("[WARN] 토픽 라벨 생성 실패:", e)

    # ───────────────────────────────
    #   Plot
    # ───────────────────────────────
    plt.figure(figsize=(14, 5))
    mean_topics.plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("평균 One-hot 비율")
    plt.title("이상치 TOP 20에서 자주 등장한 토픽 (키워드 기반)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_topic_distribution_labeled.png")
    plt.close()


# ============================
# 4) Top 20 날짜 & 점수 표 (PPT용)
# ============================
def plot_summary_table(df):
    tops = df.nlargest(20, "anomaly_score")[["date", "anomaly_score"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    table = ax.table(
        cellText=tops.values,
        colLabels=["날짜", "Anomaly Score"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.title("이상치 Top 20 (날짜 & 점수)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_top20_table.png")
    plt.close()


# ============================
# 5) 변동성 변화 + 이상치 표시
# ============================
def plot_volatility_analysis(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["volatility_10d"], label="10일 롤링 변동성")

    tops = df.nlargest(20, "anomaly_score")
    plt.scatter(tops["date"], tops["volatility_10d"], color="red", label="이상치 TOP 20")

    plt.title("변동성 변화와 이상치")
    plt.xlabel("날짜")
    plt.ylabel("10일 변동성")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_volatility_analysis.png")
    plt.close()


# ============================
# MAIN
# ============================
def generate_all():
    df = load_data()
    print("[INFO] Data Loaded")
    print("Generating charts...")

    plot_price_with_anomalies(df)
    plot_anomaly_scores(df)
    plot_topic_distribution(df)
    plot_summary_table(df)
    plot_volatility_analysis(df)

    print("[DONE] Charts saved to:", FIG_DIR)


if __name__ == "__main__":
    generate_all()
