# scripts/train_v2.py
import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.preprocessing import StandardScaler

from utils.dataset_v2 import NewsKOSPIFusionDatasetV2
from models.fusion_autoencoder_v2 import FusionAutoEncoderV2


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def train_autoencoder_v2(
    merged_path: str = "../data/processed/merged_kospi_news_v2.csv",
    scaler_path: str = "../models/scaler_v2.pkl",
    model_path: str = "../models/ae_model_v2.pt",
    emb_cache_path: str = "../data/processed/kobert_emb_v2.pt",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
):
    print("\n========================")
    print("  AutoEncoder Training  ")
    print("        (v2)           ")
    print("========================\n")

    torch.autograd.set_detect_anomaly(False)

    # ------------------------
    # Load merged dataset
    # ------------------------
    print("[STEP] Loading merged dataset (v2)...")
    df = pd.read_csv(merged_path, encoding="utf-8-sig")
    print(f" - shape: {df.shape}")
    print(f" - date range: {df['날짜'].min()} ~ {df['날짜'].max()}")

    numeric_cols = ["종가", "return", "volatility", "뉴스개수", "고유제목수"]

    # ------------------------
    # StandardScaler 생성 후 저장
    # ------------------------
    print("[STEP] Fitting StandardScaler (v2)...")

    scaler = StandardScaler()
    scaler.fit(df[numeric_cols].astype(float).values)

    ensure_dir(scaler_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[SAVE] StandardScaler (v2) → {scaler_path} (size: {os.path.getsize(scaler_path)} bytes)")

    # ------------------------
    # Dataset
    # ------------------------
    print("\n[STEP] Initializing Dataset (v2)...")

    dataset = NewsKOSPIFusionDatasetV2(
        csv_path=merged_path,
        numeric_cols=tuple(numeric_cols),
        emb_cache_path=emb_cache_path,
        scaler=scaler,
        device=device,
    )

    print("[DatasetV2] Loaded data")
    print(f" - shape: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------------------
    # Model
    # ------------------------
    print("\n[STEP] Initializing Model (v2)...")

    sample = dataset[0]
    input_dim = sample["features"].shape[-1]
    print(f"[INFO] Input Feature Dimension: {input_dim}")

    model = FusionAutoEncoderV2(
        input_dim=input_dim,
        bottleneck_dim=64,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n[INFO] Training Start! (v2)\n")

    # ------------------------
    # Training loop
    # ------------------------
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        print(f"----- Epoch {epoch}/{epochs} -----")

        for batch_idx, batch in enumerate(loader):
            feats = batch["features"].to(device)

            optimizer.zero_grad()
            recon, _ = model(feats)
            loss = criterion(recon, feats)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  [Batch {batch_idx+1}/{len(loader)}] Loss: {loss.item():.6f}")

        print(f"[Epoch {epoch:02d}] Avg Loss = {epoch_loss / len(loader):.6f}\n")

    print("[INFO] Training Complete! (v2)")

    # 저장
    ensure_dir(model_path)
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Saved model (v2) → {model_path}")


if __name__ == "__main__":
    train_autoencoder_v2(device="cpu")
