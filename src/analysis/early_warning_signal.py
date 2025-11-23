# src/analysis/early_warning_signal.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("C:\\MyProject\\PytorchProject\\data\\processed\\anomaly_scores.parquet")

df["rolling_mean"] = df["anomaly_score"].rolling(20).mean()
df["rolling_std"] = df["anomaly_score"].rolling(20).std()

df["warning"] = df["anomaly_score"] > (df["rolling_mean"] + 2*df["rolling_std"])

sns.lineplot(data=df, x="date", y="anomaly_score", label="AE score")
sns.scatterplot(data=df[df["warning"]], x="date", y="anomaly_score", color="red", label="Warning")
plt.show()
