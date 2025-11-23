# scripts/inference.py

import torch
from torch.utils.data import DataLoader
import pandas as pd

from utils.dataset import NewsStockDataset
from models.fusion_autoencoder import FusionAutoEncoder


def run_inference(
    csv_path="../data/processed/merged_kospi_news.csv",
    model_path="../models/ae_model.pt",
    batch_size=32,
    device="cpu",
):
    print("\n========================")
    print("   AutoEncoder Infer    ")
    print("========================\n")

    # ------------------------
    # 1) Dataset & DataLoader
    # ------------------------
    print("[STEP] Loading Dataset for inference...")

    dataset = NewsStockDataset(
        csv_path=csv_path,
        numeric_cols=("종가", "return", "volatility"),
        # 캐시를 쓰면 다음부터는 빠르게 동작
        use_cached_emb=True,
        emb_save_path="../data/processed/kobert_emb.pt",
        device=device,
    )

    print(f"[INFO] 전체 데이터 수: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] DataLoader batch_size={batch_size}, total_batches={len(loader)}\n")

    # ------------------------
    # 2) Model 복원
    # ------------------------
    print("[STEP] Loading Trained Model...")

    sample = dataset[0]
    input_dim = sample["features"].shape[-1]
    print(f"[INFO] Input Feature Dimension: {input_dim}")

    model = FusionAutoEncoder(
        input_dim=input_dim,
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[INFO] Loaded model from: {model_path}")
    print(f"[INFO] Model on device: {device}\n")

    # ------------------------
    # 3) 각 샘플별 재구성 오차 계산
    # ------------------------
    print("[STEP] Computing reconstruction loss per sample...\n")

    all_losses = []
    all_dates = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features = batch["features"].to(device)
            recon, _ = model(features)

            # (batch, feature_dim) → 각 row별 MSE
            batch_loss = ((recon - features) ** 2).mean(dim=1)  # (batch,)

            all_losses.extend(batch_loss.cpu().tolist())
            all_dates.extend(batch["date"])  # 날짜 그대로 모으기

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(f"  [Batch {batch_idx+1}/{len(loader)}] "
                      f"mean loss: {batch_loss.mean().item():.4f}")

    all_losses = torch.tensor(all_losses)
    print("\n[INFO] 전체 샘플 기준 통계")
    print(f" - 개수: {all_losses.numel()}")
    print(f" - 평균: {all_losses.mean().item():.4f}")
    print(f" - 표준편차: {all_losses.std().item():.4f}")
    print(f" - 최소: {all_losses.min().item():.4f}")
    print(f" - 최대: {all_losses.max().item():.4f}")

    # ------------------------
    # 4) 원본 CSV에 anomaly score 붙이기
    # ------------------------
    print("\n[STEP] Save anomaly scores to CSV...")

    df = pd.read_csv(csv_path)
    if len(df) != len(all_losses):
        raise ValueError(
            f"Data length mismatch! df: {len(df)}, losses: {len(all_losses)}"
        )

    df["recon_loss"] = all_losses.numpy()
    # 날짜는 이미 merged_kospi_news.csv에 들어있으므로 그대로 둠

    save_path = "../data/processed/anomaly_scores.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[SUCCESS] Saved anomaly scores → {save_path}")


if __name__ == "__main__":
    run_inference(device="cpu")
