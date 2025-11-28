# models/train_ae_622.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR, AE_MODEL_DIR
from models.autoencoder import AutoEncoder


def train_622(device="cpu", epochs=50, batch=64, lr=1e-3):
    data = torch.load(PROCESSED_DIR / "multimodal_dataset_split622.pt", weights_only=False)

    X_train = data["X_train"]
    X_val = data["X_val"]

    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim).to(device)

    dl_train = DataLoader(TensorDataset(X_train), batch_size=batch, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val), batch_size=batch, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    best_val = 1e10
    save_path = AE_MODEL_DIR / "autoencoder622_best.pt"

    for epoch in range(1, epochs + 1):

        # ---- Training ----
        model.train()
        total_train = 0
        for (xb,) in dl_train:
            xb = xb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            total_train += loss.item() * xb.size(0)
        train_loss = total_train / len(X_train)

        # ---- Validation ----
        model.eval()
        total_val = 0
        with torch.no_grad():
            for (xb,) in dl_val:
                xb = xb.to(device)
                out = model(xb)
                loss = loss_fn(out, xb)
                total_val += loss.item() * xb.size(0)
        val_loss = total_val / len(X_val)

        print(f"[EPOCH {epoch:03}] train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim}, save_path)
            print(" → Best model saved:", save_path)

        history.append([epoch, train_loss, val_loss])

    # ---- Save Log ----
    df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"])
    df_path = PROCESSED_DIR / "ae_training_history_622.csv"
    df.to_csv(df_path, index=False)

    # ---- Save Curve ----
    plt.figure(figsize=(7,4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("AutoEncoder Training Curve (6:2:2)")
    plt.tight_layout()

    fig_path = AE_MODEL_DIR / "ae_training_curve_64.png"
    plt.savefig(fig_path)
    plt.close()

    print("✔ Log:", df_path)
    print("✔ Figure:", fig_path)


if __name__ == "__main__":
    train_622(device="cpu")
