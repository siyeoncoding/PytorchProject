# src/features/build_dataset.py
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import INTERIM_DIR, EMBEDDING_DIR, PROCESSED_DIR

def build():
    df = pd.read_parquet(INTERIM_DIR / "timeseries_with_sentiment_topic.parquet")
    emb = torch.load(EMBEDDING_DIR / "kobert_embeddings.pt")

    num_cols = ["close","return","log_return","volatility_10d","volume","news_count"]
    num_data = df[num_cols].fillna(0).values

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_data)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)

    sent_tensor = torch.tensor(df["sentiment"].values, dtype=torch.float32).unsqueeze(1)

    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    topic_tensor = torch.tensor(df[topic_cols].values, dtype=torch.float32)

    X = torch.cat([num_tensor, sent_tensor, topic_tensor, emb.float()], dim=1)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    split = pd.to_datetime("2025-01-01").date()

    train_idx = df["date"] < split
    test_idx = df["date"] >= split

    out = {
        "X_train": X[train_idx.values],
        "X_test": X[test_idx.values],
        "dates_train": df.loc[train_idx,"date"].tolist(),
        "dates_test": df.loc[test_idx,"date"].tolist(),
        "scaler": scaler
    }

    torch.save(out, PROCESSED_DIR / "multimodal_dataset.pt")

if __name__ == "__main__":
    build()
