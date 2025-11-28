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
        text = str(row.get("merged_text", "")).strip()

        # 너무 길면 잘라서 사용 (토크나이저 오버플로우 방지)
        text = text[:2048]

        log(f"[{i}/20] summarizing {date} (score={score:.6f}, close={close:.2f})")

        if not text:
            summaries.append("해당 날짜에 요약 가능한 뉴스 데이터가 부족함")
            continue

        try:
            # max_new_tokens만 사용해서 경고 줄이기
            result = summarizer(
                text,
                max_new_tokens=80,
                do_sample=False,
                truncation=True,
            )
            summ = result[0]["summary_text"].strip()
        except Exception as e:
            log(f"  summarization failed: {e}")
            summ = "요약 생성 실패"

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
    plt.savefig(out_path, dpi=220)
    plt.close()
    log(f"Saved figure: {out_path}")


def _plot_summary_table_single(df_tbl: pd.DataFrame, part_idx: int, total_parts: int):
    """
    내부용: df_tbl(최대 10행)을 받아서 하나의 표 이미지를 저장한다.
    part_idx: 1 또는 2
    total_parts: 전체 파트 개수(여기서는 2)
    """
    # 요약을 여러 줄로 줄바꿈해서 보기 좋게 만들기
    def wrap_multiline(s, width=70):
        return "\n".join(textwrap.wrap(str(s), width=width))

    df_tbl = df_tbl.copy()
    df_tbl["summary_wrapped"] = df_tbl["summary"].apply(wrap_multiline)

    display_cols = ["date", "close", "anomaly_score", "summary_wrapped"]
    col_labels = ["날짜", "종가", "Anomaly Score", "뉴스 요약(요약본)"]

    # 그림 크게, 요약 칸 넓게
    fig, ax = plt.subplots(figsize=(26, 12), dpi=220)
    ax.axis("off")

    table = ax.table(
        cellText=df_tbl[display_cols].values,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.07, 0.07, 0.09, 0.77],  # 뉴스 요약 칸 가장 넓게
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    plt.title(f"이상치 Top 20 날짜별 뉴스 요약 (Part {part_idx}/{total_parts})", pad=20)
    plt.tight_layout()

    out_path = FIG_DIR / f"A2_top20_anomaly_summaries_part{part_idx}.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    log(f"Saved figure: {out_path}")


def plot_top20_summary_table(top):
    """
    Top20 날짜 + 점수 + 요약을
    10개씩 나눠서 두 장의 표 이미지로 저장
    """
    log("Plotting Top20 summary tables (split into 2 parts)...")

    # 안전하게 정렬
    top_sorted = top.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # 상위 10개 / 하위 10개로 분할
    part1 = top_sorted.iloc[:10]
    part2 = top_sorted.iloc[10:]

    _plot_summary_table_single(part1, part_idx=1, total_parts=2)
    _plot_summary_table_single(part2, part_idx=2, total_parts=2)


def main():
    df = load_anomaly_data()
    top = get_top20_with_summary(df)

    # 그래프 1: 가격 + 이상치
    plot_price_with_anomalies(df, top)

    # 그래프 2: Top20 요약 표 (10개씩 2장)
    plot_top20_summary_table(top)

    log("Done. Figures ready for PPT:")
    print(" - A1_price_with_anomalies_top20.png")
    print(" - A2_top20_anomaly_summaries_part1.png")
    print(" - A2_top20_anomaly_summaries_part2.png")


if __name__ == "__main__":
    main()
