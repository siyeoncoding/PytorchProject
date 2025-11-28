# src/models/train_ae.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR, AE_MODEL_DIR
from models.autoencoder import AutoEncoder


def train(device="cpu", epochs=50, batch=64, lr=1e-3):
    # 1) 데이터 로드 (train/val 함께 사용)
    data = torch.load(PROCESSED_DIR / "multimodal_dataset.pt", weights_only=False)

    X_train = data["X_train"]
    X_val = data["X_test"]      # 보고서에서는 "검증(Validation) 셋"으로 사용
    input_dim = X_train.shape[1]

    # 2) 모델/데이터로더 준비
    model = AutoEncoder(input_dim).to(device)

    train_ds = TensorDataset(X_train)
    val_ds = TensorDataset(X_val)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    save_path = AE_MODEL_DIR / "autoencoder_best.pt"

    # 로그를 저장할 리스트
    history = []

    for epoch in range(1, epochs + 1):
        # ------------------------------
        # 3) Train step
        # ------------------------------
        model.train()
        total_train = 0.0

        for (xb,) in train_dl:
            xb = xb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()

            total_train += loss.item() * xb.size(0)

        avg_train = total_train / len(train_ds)

        # ------------------------------
        # 4) Validation step
        # ------------------------------
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(device)
                out = model(xb)
                loss = loss_fn(out, xb)
                total_val += loss.item() * xb.size(0)

        avg_val = total_val / len(val_ds)

        # ------------------------------
        # 5) 로그 출력 + Best 모델 저장
        # ------------------------------
        print(f"[EPOCH {epoch:03d}] train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
            }
        )

        # validation 기준으로 best 모델 저장
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {"model_state": model.state_dict(), "input_dim": input_dim},
                save_path,
            )
            print("  → Best model updated & saved")

    # --------------------------------
    # 6) 로그를 CSV + 그래프로 저장
    # --------------------------------
    hist_df = pd.DataFrame(history)
    log_path = PROCESSED_DIR / "ae_training_history.csv"
    hist_df.to_csv(log_path, index=False)
    print(f"[LOG SAVED] {log_path}")

    # 그래프 저장 (PPT용)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Reconstruction Loss")
    plt.title("AutoEncoder Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = AE_MODEL_DIR / "ae_training_curve_64batch.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[FIGURE SAVED] {fig_path}")


if __name__ == "__main__":
    train(device="cpu")
