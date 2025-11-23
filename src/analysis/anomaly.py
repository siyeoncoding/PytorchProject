# src/analysis/analyze_top_anomalies.py

import pandas as pd
from transformers import pipeline
from src.config import PROCESSED_DIR

def analyze():
    # 1) 이상치 점수 파일 로드
    path = PROCESSED_DIR / "anomaly_scores.parquet"
    print("[LOAD]", path)
    df = pd.read_parquet(path)

    # 2) anomaly_score 기준 상위 20개 날짜 선택
    top = df.nlargest(20, "anomaly_score").copy()

    # 3) 한국어 뉴스 요약 모델 로드
    #    (요약 전용 KoBART 모델)
    summarizer = pipeline(
        "summarization",
        model="gogamza/kobart-summarization"   # 또는 eunjin/kobart-summary
    )

    # 4) 각 날짜별로 뉴스 텍스트 요약 출력
    for _, row in top.iterrows():
        date = row["date"]
        score = row["anomaly_score"]
        close = row["close"]
        text = str(row.get("merged_text", ""))[:2048]

        print("=" * 80)
        print(f"[날짜] {date} | [종가] {close:.2f} | [Anomaly Score] {score:.6f}")
        print("-" * 80)

        if not text.strip():
            print("뉴스 텍스트 없음")
            continue

        summary = summarizer(text)[0]["summary_text"]
        print("뉴스 요약:")
        print(summary)
        print()

if __name__ == "__main__":
    analyze()
