# src/models/evaluate_ae.py
import torch
import torch.nn as nn
import pandas as pd

from models.autoencoder import AutoEncoder
from src.config import PROCESSED_DIR, INTERIM_DIR, AE_MODEL_DIR


def compute_scores(device="cpu"):
    data = torch.load(PROCESSED_DIR / "multimodal_dataset.pt", weights_only=False)

    X_test = data["X_test"]
    dates_test = data["dates_test"]

    ckpt = torch.load(AE_MODEL_DIR / "autoencoder_best.pt", map_location=device)
    model = AutoEncoder(ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    loss_fn = nn.MSELoss(reduction="none")

    with torch.no_grad():
        X_test = X_test.to(device)
        recon = model(X_test)
        scores = loss_fn(recon, X_test).mean(dim=1).cpu().numpy()

    df = pd.read_parquet(INTERIM_DIR / "merged_timeseries.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df_test = df[df["date"].isin(dates_test)].copy()
    df_test["anomaly_score"] = scores

    out_path = PROCESSED_DIR / "anomaly_scores.parquet"
    df_test.to_parquet(out_path, index=False)

    print("[ANOMALY SCORES SAVED]")
    print(" →", out_path)
    return out_path


if __name__ == "__main__":
    compute_scores(device="cpu")
