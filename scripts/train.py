import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from utils.dataset import NewsStockDataset
from models.fusion_autoencoder import FusionAutoEncoder


def train_autoencoder(
    csv_path="../data/processed/merged_kospi_news.csv",
    epochs=20,
    batch_size=8,
    lr=1e-4,          # 🔽 우선 러닝레이트도 살짝 낮춰보자
    device="cpu",
):
    print("\n========================")
    print("  AutoEncoder Training  ")
    print("========================\n")

    # 디버깅용
    torch.autograd.set_detect_anomaly(True)

    # ------------------------
    # Dataset & DataLoader
    # ------------------------
    print("[STEP] Loading Dataset...")
    dataset = NewsStockDataset(
        csv_path=csv_path,
        numeric_cols=("종가", "return", "volatility"),
        use_cached_emb=False,
        device=device,
    )

    print(f"[INFO] 전체 데이터 수: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[INFO] DataLoader batch_size={batch_size}, total_batches={len(loader)}\n")

    # ------------------------
    # Model
    # ------------------------
    print("[STEP] Initializing Model...")

    sample = dataset[0]
    input_dim = sample["features"].shape[-1]
    print(f"[INFO] Input Feature Dimension: {input_dim}")

    model = FusionAutoEncoder(
        input_dim=input_dim,
        hidden_dim=128,
        dropout=0.1
    ).to(device)

    print(f"[INFO] Model moved to device: {device}\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------------
    # Training Loop
    # ------------------------
    print("[INFO] Training Start!\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        print(f"----- Epoch {epoch}/{epochs} -----")

        for batch_idx, batch in enumerate(loader):
            features = batch["features"].to(device)

            # 🔍 1) 입력에 NaN 있는지 확인
            if torch.isnan(features).any():
                print(f"[ERROR] features에 NaN 발생! epoch={epoch}, batch={batch_idx}")
                print(" - NaN 포함 row index 예시:", (torch.isnan(features).any(dim=1).nonzero(as_tuple=True)[0][:10]))
                return

            optimizer.zero_grad()
            recon, _ = model(features)

            # 🔍 2) 출력에도 NaN 있는지 확인
            if torch.isnan(recon).any():
                print(f"[ERROR] recon에 NaN 발생! epoch={epoch}, batch={batch_idx}")
                return

            loss = criterion(recon, features)

            # 🔍 3) Loss 자체가 NaN 인지 확인
            if torch.isnan(loss):
                print(f"[ERROR] Loss가 NaN! epoch={epoch}, batch={batch_idx}")
                return

            loss.backward()

            # 🔧 4) gradient 폭발 방지용 clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx == 0:
                print(f"[DEBUG] Batch 0 features shape: {features.shape}")
            if (batch_idx + 1) % 20 == 0:
                print(f"  [Batch {batch_idx+1}/{len(loader)}] Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch:02d}] Average Loss: {avg_loss:.6f}\n")

    print("[INFO] Training Complete!\n")

    save_path = "../models/ae_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Saved model → {save_path}")


if __name__ == "__main__":
    train_autoencoder(device="cpu")
