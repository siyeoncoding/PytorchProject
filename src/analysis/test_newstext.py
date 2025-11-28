import pandas as pd

df = anomaly_df.copy()
df["abs_return"] = df["return"].abs()

# Reconstruction Error와 실제 변동률 상관관계
corr = df["anomaly_score"].corr(df["abs_return"])

print("상관계수:", corr)

df["is_spike"] = (
    (df["abs_return"] > 0.01) |
    (df["volatility_10d"] > df["volatility_10d"].mean() + 2 * df["volatility_10d"].std()) |
    (df["volume"] > df["volume"].mean() + 2 * df["volume"].std())
)

# 이상치 중 실제 이벤트였던 비율
accuracy = (df[df["is_anomaly"]]["is_spike"].mean())

print("이상치가 실제 급등/급락 시점과 일치한 비율:", accuracy)

