# src/models/train_ae.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROCESSED_DIR, AE_MODEL_DIR
from models.autoencoder import AutoEncoder

def train(device="cpu", epochs=50, batch=32, lr=1e-3):
    data = torch.load(PROCESSED_DIR / "multimodal_dataset.pt", weights_only=False)

    X_train = data["X_train"]
    input_dim = X_train.shape[1]

    model = AutoEncoder(input_dim).to(device)
    ds = TensorDataset(X_train)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = 1e10
    save_path = AE_MODEL_DIR / "autoencoder_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0

        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        avg = total / len(ds)
        print(f"[EPOCH {epoch}] loss={avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim}, save_path)
            print(" → Model saved")


if __name__ == "__main__":
    train(device="cpu")
