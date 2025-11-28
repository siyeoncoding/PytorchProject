import re
import textwrap
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

from src.config import PROCESSED_DIR

# -------------------------------------------------
# 0) 폰트 및 스타일 설정
# -------------------------------------------------
# Windows: Malgun Gothic, Mac: AppleGothic 등 환경에 맞게 설정
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[INFO] {msg}")


# -------------------------------------------------
# 1) 데이터 로드
# -------------------------------------------------
def load_anomaly_data():
    path = PROCESSED_DIR / "anomaly_scores.parquet"
    log(f"Loading anomaly scores from: {path}")
    df = pd.read_parquet(path)
    df = df.sort_values("date")
    log(f"Loaded {len(df)} rows")
    return df


# -------------------------------------------------
# 2) 뉴스 앞문장 추출 함수
# -------------------------------------------------
def simple_extract(text: str, max_sent: int = 2, max_chars: int = 200) -> str:
    """
    긴 뉴스 텍스트에서 앞쪽 문장 1~2개만 잘라서 사용
    """
    if text is None:
        return "-"

    text = str(text).strip()
    if not text:
        return "-"

    # 문장 단위로 분리 (마침표/물음표/느낌표/줄바꿈 기준)
    sents = re.split(r"(?<=[\.!?…])\s+|\n+", text)
    sents = [s.strip() for s in sents if s.strip()]

    if not sents:
        return "-"

    summary = " ".join(sents[:max_sent])

    # 너무 길면 자르기
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."

    return summary


# -------------------------------------------------
# 3) Top20 이상치 + 앞문장 추출
# -------------------------------------------------
def get_top20_with_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    상위 20개 이상치 날짜를 선택하고,
    각 날짜의 merged_text에서 앞문장 1~2개만 추출해서 summary 컬럼에 저장
    """
    top = df.nlargest(20, "anomaly_score").copy()

    summaries = []
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        date = row["date"]
        text = row.get("merged_text", "")
        # 추출 (최대 길이 제한을 조금 줄여서 칸 넘침 방지)
        summaries.append(simple_extract(text, max_sent=2, max_chars=180))

    top["summary"] = summaries
    return top


# -------------------------------------------------
# 4) 가격 + 이상치 그래프
# -------------------------------------------------
def plot_price_with_anomalies(df: pd.DataFrame, top: pd.DataFrame):
    """KOSPI 가격 + 이상치 위치 표시 그래프"""
    log("Plotting price with anomalies...")

    plt.figure(figsize=(14, 6))

    # 메인 라인
    plt.plot(df["date"], df["close"], label="KOSPI 종가", color="#1f77b4", linewidth=1.5)

    # 이상치 포인트
    plt.scatter(
        top["date"], top["close"],
        color="#d62728", s=80, zorder=5,
        edgecolors="white", linewidth=1.5, label="이상치 TOP 20"
    )

    plt.title("KOSPI 가격 추이 및 주요 이상 탐지 시점 (Top 20)", fontsize=16, pad=15, fontweight="bold")
    plt.xlabel("날짜", fontsize=12)
    plt.ylabel("종가", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = FIG_DIR / "A1_price_with_anomalies_top20.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved figure: {out_path}")


# -------------------------------------------------
# 5) 표 한 장 그리는 내부 함수 (디자인 개선)
# -------------------------------------------------
def _plot_summary_table_single(df_tbl: pd.DataFrame, part_idx: int, total_parts: int):
    """
    가독성을 위해 디자인을 대폭 개선한 테이블 그리기 함수
    """
    df_draw = df_tbl.copy()

    # 1. 데이터 포맷팅
    df_draw["date_str"] = df_draw["date"].astype(str).str[:10]
    df_draw["close_str"] = df_draw["close"].apply(lambda x: f"{x:,.0f}")
    df_draw["score_str"] = df_draw["anomaly_score"].apply(lambda x: f"{x:.4f}")

    # [수정] 뉴스 텍스트 래핑 너비 증가 (38 -> 64)
    # 가로 공간을 더 많이 써서 줄바꿈 횟수를 줄임 -> 세로 겹침 완화
    def wrap_multiline(s, width=64):
        return "\n".join(textwrap.wrap(str(s), width=width))

    df_draw["summary_wrapped"] = df_draw["summary"].apply(wrap_multiline)

    display_data = df_draw[["date_str", "close_str", "score_str", "summary_wrapped"]].values
    col_labels = ["날짜", "종가", "이상치 점수", "주요 뉴스 요약"]

    # -------------------------------------------
    # 그림 생성
    # -------------------------------------------
    fig_height = 12
    fig, ax = plt.subplots(figsize=(20, fig_height), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=display_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.1, 0.1, 0.12, 0.68]
    )

    # -------------------------------------------
    # 스타일링 (핵심)
    # -------------------------------------------
    table.auto_set_font_size(False)
    # [수정] 폰트 크기 살짝 축소하여 여유 확보 (13 -> 12)
    table.set_fontsize(12)

    # [수정] 행 높이(Scale) 추가 확대 (3.8 -> 4.5)
    # 줄바꿈된 텍스트가 위아래로 넉넉하게 들어가도록 함
    table.scale(1.0, 4.5)

    # 셀 별 스타일 적용
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(1.0)

        # 헤더 스타일 (Row 0)
        if row == 0:
            cell.set_text_props(weight="bold", color="white", ha="center", va="center")
            cell.set_facecolor("#404040")
            cell.set_height(0.15)
            cell.set_fontsize(14)
        else:
            # 데이터 셀 스타일
            cell.set_text_props(va="center")

            if col < 3:
                cell.set_text_props(ha="center", va="center")
            else:
                # 뉴스 컬럼
                cell.set_text_props(ha="left", va="center")

            # 지브라 패턴
            if row % 2 == 0:
                cell.set_facecolor("#F8F9FA")
            else:
                cell.set_facecolor("#FFFFFF")

    plt.title(f"이상치 Top 20 분석 리포트 (Page {part_idx}/{total_parts})",
              fontsize=18, fontweight="bold", pad=20, color="#333333")

    plt.tight_layout()

    out_path = FIG_DIR / f"A2_top20_anomaly_summaries_part{part_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved table figure: {out_path}")


# -------------------------------------------------
# 6) Top20을 10개씩 2장으로 나눠서 그리기
# -------------------------------------------------
def plot_top20_summary_table(top: pd.DataFrame):
    """
    Top20 날짜 + 점수 + 앞문장 요약을
    10개씩 나눠서 두 장의 표 이미지로 저장
    """
    log("Plotting Top20 summary tables (split into 2 parts)...")

    # anomaly_score 기준으로 내림차순 정렬
    top_sorted = top.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # 10개씩 분할
    part1 = top_sorted.iloc[:10]
    part2 = top_sorted.iloc[10:]

    _plot_summary_table_single(part1, part_idx=1, total_parts=2)

    if not part2.empty:
        _plot_summary_table_single(part2, part_idx=2, total_parts=2)


# -------------------------------------------------
# 7) main
# -------------------------------------------------
def main():
    df = load_anomaly_data()

    if df.empty:
        log("No anomaly data found.")
        return

    top = get_top20_with_summary(df)

    if top.empty:
        log("No top anomalies found.")
        return

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