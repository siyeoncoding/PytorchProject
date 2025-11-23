# src/analysis/volatility_relation.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("C:\\MyProject\\PytorchProject\\data\\processed\\anomaly_scores.parquet")

df["future_vol"] = df["volatility_10d"].shift(-3)

sns.scatterplot(data=df, x="anomaly_score", y="future_vol")
plt.title("AE 오차 vs 3일 후 변동성")
plt.show()
