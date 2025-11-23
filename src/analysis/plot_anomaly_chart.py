# src/analysis/plot_anomaly_chart.py
import pandas as pd
import plotly.graph_objects as go

df = pd.read_parquet("C:\\MyProject\\PytorchProject\\data\\processed\\anomaly_scores.parquet")
thr = df["anomaly_score"].quantile(0.95)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="KOSPI"))

ano = df[df["anomaly_score"]>thr]
fig.add_trace(go.Scatter(x=ano["date"], y=ano["close"], mode="markers",
                         marker=dict(color="red",size=8), name="Anomaly"))

fig.show()
