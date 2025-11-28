# src/analysis/check_anomaly_alignment.py
"""
Reconstruction Error 기반 이상치가
실제 급등·급락(가격 스파이크)과 얼마나 일치하는지 평가하는 스크립트.

1) anomaly_score ↔ |return| 상관계수
2) 이상치(Top K) 중에서 실제 스파이크 비율
"""

import pandas as pd

from src.config import PROCESSED_DIR, INTERIM_DIR


# -----------------------------
# 1. 데이터 로드 & 병합
# -----------------------------
def load_anomaly_dataframe(
    merged_path: str | None = None,
    scores_path: str | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    merged_timeseries + anomaly_scores를 병합해서
    이상치 분석에 필요한 컬럼만 가진 DataFrame을 반환.

    필요 컬럼:
      - date
      - close
      - return
      - volatility_10d
      - volume
      - anomaly_score
      - is_anomaly(Top K)
    """

    if merged_path is None:
        merged_path = INTERIM_DIR / "merged_timeseries.parquet"
    if scores_path is None:
        # 네가 사용한 파일명에 맞게 변경해도 됨
        scores_path = PROCESSED_DIR / "anomaly_scores.parquet"

    print(f"[LOAD] merged_timeseries: {merged_path}")
    base = pd.read_parquet(merged_path)

    print(f"[LOAD] anomaly_scores:   {scores_path}")
    scores = pd.read_parquet(scores_path)

    # date 컬럼 타입 맞추기 (문자/날짜 혼재 방지)
    base["date"] = pd.to_datetime(base["date"]).dt.date
    scores["date"] = pd.to_datetime(scores["date"]).dt.date

    # inner join: 공통 날짜만 사용
    df = pd.merge(
        base,
        scores[["date", "anomaly_score"]],
        on="date",
        how="inner",
    )

    # Top K 이상치 플래그
    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    df["is_anomaly"] = False
    df.loc[: top_k - 1, "is_anomaly"] = True

    print(f"[INFO] merged df shape: {df.shape}")
    print(f"[INFO] Top {top_k} rows flagged as anomaly.")
    return df


# -----------------------------
# 2. 상관 분석 + 스파이크 일치도 평가
# -----------------------------
def evaluate_alignment(df: pd.DataFrame) -> None:
    """
    Reconstruction Error 기반 이상치가
    실제 급등·급락 시점과 얼마나 맞는지 수치로 평가.
    """

    # 절대 수익률 컬럼
    df = df.copy()
    df["abs_return"] = df["return"].abs()

    # 1) 이상치 점수 vs 절대 수익률 상관계수
    corr = df["anomaly_score"].corr(df["abs_return"])
    print("\n[METRIC] anomaly_score ↔ |return| 상관계수:", round(corr, 4))

    # 2) '실제 이벤트(스파이크)' 정의
    #    - |return| > 1%
    #    - 변동성이 평균 + 2σ 초과
    #    - 거래량이 평균 + 2σ 초과
    vol_th = df["volatility_10d"].mean() + 2 * df["volatility_10d"].std()
    volm_th = df["volume"].mean() + 2 * df["volume"].std()

    df["is_spike"] = (
        (df["abs_return"] > 0.01)
        | (df["volatility_10d"] > vol_th)
        | (df["volume"] > volm_th)
    )

    # 3) 이상치(Top K) 중에서 실제 스파이크 비율
    anomaly_df = df[df["is_anomaly"]]
    if len(anomaly_df) == 0:
        print("[WARN] is_anomaly=True 인 데이터가 없습니다.")
        return

    accuracy = anomaly_df["is_spike"].mean()
    print(
        "[METRIC] 이상치가 실제 급등/급락(스파이크) 시점과 일치한 비율:",
        f"{accuracy * 100:.2f}%",
    )

    # 4) 참고용: 이상치 Top K 테이블 간단 출력
    print("\n[TOP ANOMALY SAMPLE]")
    print(
        anomaly_df[
            ["date", "close", "return", "volatility_10d", "volume", "anomaly_score", "is_spike"]
        ].head(20)
    )


# -----------------------------
# 3. main
# -----------------------------
def main():
    # 필요하면 top_k, 파일 경로를 직접 바꿔도 됨
    df = load_anomaly_dataframe(
        merged_path=INTERIM_DIR / "merged_timeseries.parquet",
        scores_path=PROCESSED_DIR / "anomaly_scores.parquet",
        top_k=20,  # Reconstruction Error 상위 20개를 이상치로 사용
    )

    evaluate_alignment(df)


if __name__ == "__main__":
    main()
