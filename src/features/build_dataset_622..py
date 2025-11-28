# src/features/build_dataset_622.py

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import INTERIM_DIR, EMBEDDING_DIR, PROCESSED_DIR


def build_622():
    df = pd.read_parquet(INTERIM_DIR / "timeseries_with_sentiment_topic.parquet")
    emb = torch.load(EMBEDDING_DIR / "kobert_embeddings.pt")

    # ---- 수치형 피처 ----
    num_cols = ["close","return","log_return","volatility_10d","volume","news_count"]
    num_data = df[num_cols].fillna(0).values

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_data)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)

    # ---- 감성 ----
    sent_tensor = torch.tensor(df["sentiment"].values, dtype=torch.float32).unsqueeze(1)

    # ---- 토픽 ----
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    topic_tensor = torch.tensor(df[topic_cols].values, dtype=torch.float32)

    # ---- KoBERT 임베딩 ----
    X = torch.cat([num_tensor, sent_tensor, topic_tensor, emb.float()], dim=1)

    # ---- 날짜 정렬 ----
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    total_len = len(df)
    t1 = int(total_len * 0.6)
    t2 = int(total_len * 0.8)

    idx_train = list(range(0, t1))
    idx_val   = list(range(t1, t2))
    idx_test  = list(range(t2, total_len))

    out = {
        "X_train": X[idx_train],
        "X_val":   X[idx_val],
        "X_test":  X[idx_test],
        "dates_train": df.loc[idx_train, "date"].tolist(),
        "dates_val":   df.loc[idx_val, "date"].tolist(),
        "dates_test":  df.loc[idx_test, "date"].tolist(),
        "scaler": scaler
    }

    save_path = PROCESSED_DIR / "multimodal_dataset_split622.pt"
    torch.save(out, save_path)

    print("✔ Saved:", save_path)


if __name__ == "__main__":
    build_622()
