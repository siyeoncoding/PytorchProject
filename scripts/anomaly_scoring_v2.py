# scripts/anomaly_scoring_v2.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn

from utils.dataset_v2 import NewsKOSPIFusionDatasetV2
from models.fusion_autoencoder_v2 import FusionAutoEncoderV2


def compute_thresholds(scores: np.ndarray, top_percent: float = 5.0, mad_k: float = 3.0):
    """
    recon_loss 배열(scores)을 받아서
    - TOP N% (상위 퍼센타일)
    - IQR (Q3 + 1.5*IQR)
    - MAD (median + k * MAD)
    기준 threshold를 계산해서 리턴
    """
    scores = np.asarray(scores, dtype=float)

    # TOP N% 기준
    top_thr = np.quantile(scores, 1.0 - top_percent / 100.0)

    # IQR 기준 (Q3 + 1.5*IQR)
    q1 = np.quantile(scores, 0.25)
    q3 = np.quantile(scores, 0.75)
    iqr = q3 - q1
    iqr_thr = q3 + 1.5 * iqr

    # MAD 기준 (median + k * MAD)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mad_thr = median + mad_k * mad

    thresholds = {
        "topN": top_thr,
        "iqr": iqr_thr,
        "mad": mad_thr,
        "q1": q1,
        "q3": q3,
        "iqr_raw": iqr,
        "median": median,
        "mad_raw": mad,
    }
    return thresholds


def anomaly_scoring_v2(
    merged_path: str = "../data/processed/merged_kospi_news_v2.csv",
    model_path: str = "../models/ae_model_v2.pt",
    emb_cache_path: str = "../data/processed/kobert_emb_v2.pt",
    save_path: str = "../data/processed/anomaly_scores_v2.csv",
    device: str = "cpu",
    top_percent: float = 5.0,
    mad_k: float = 3.0,
    batch_size: int = 32,
):
    print("\n========================")
    print("  Anomaly Scoring  (v2) ")
    print("========================\n")

    # ------------------------
    # 1) 데이터 로드
    # ------------------------
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"merged file not found: {merged_path}")

    df = pd.read_csv(merged_path, encoding="utf-8-sig")
    print("[STEP] Load merged_kospi_news_v2.csv")
    print(f" - shape: {df.shape}")
    print(f" - date range: {df['날짜'].min()} ~ {df['날짜'].max()}")

    # 숫자형 컬럼 확인 (뉴스개수 / 고유제목수 포함)
    numeric_cols = ["종가", "return", "volatility", "뉴스개수", "고유제목수"]
    for col in numeric_cols:
        if col not in df.columns:
            raise KeyError(f"Numeric column '{col}' not found in merged file.")

    # ------------------------
    # 2) Dataset / DataLoader
    # ------------------------
    print("\n[STEP] Initializing Dataset (v2)...")

    # ⚠️ dataset_v2 시그니처에 맞게 인자 정리
    dataset = NewsKOSPIFusionDatasetV2(
        csv_path=merged_path,
        numeric_cols=tuple(numeric_cols),
        emb_cache_path=emb_cache_path,
        device=device,
    )

    print(f"[INFO] 전체 샘플 수: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,   # ⚠ 순서 유지 (row ↔ loss 매칭용)
    )
    print(f"[INFO] DataLoader batch_size={batch_size}, total_batches={len(loader)}")

    # ------------------------
    # 3) 모델 로드
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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[INFO] Loaded model (v2) from: {model_path}")
    print(f"[INFO] Model on device: {device}")

    criterion = nn.MSELoss(reduction="none")

    # ------------------------
    # 4) reconstruction loss per sample
    # ------------------------
    print("\n[STEP] Computing reconstruction loss per sample (v2)...\n")

    all_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features = batch["features"].to(device)  # (B, D)
            recon, _ = model(features)              # (B, D)

            # sample-wise MSE (평균)
            loss_per_feature = criterion(recon, features)  # (B, D)
            loss_per_sample = loss_per_feature.mean(dim=1)  # (B,)

            all_losses.append(loss_per_sample.cpu().numpy())

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"  [Batch {batch_idx+1}/{len(loader)}] "
                    f"mean recon_loss: {loss_per_sample.mean().item():.4f}"
                )

    all_losses = np.concatenate(all_losses, axis=0)
    if len(all_losses) != len(df):
        raise RuntimeError(
            f"recon_loss length mismatch: len(losses)={len(all_losses)}, len(df)={len(df)}"
        )

    df["recon_loss"] = all_losses

    # ------------------------
    # 5) Threshold 계산 (TOP N%, IQR, MAD)
    # ------------------------
    thresholds = compute_thresholds(all_losses, top_percent=top_percent, mad_k=mad_k)

    top_thr = thresholds["topN"]
    iqr_thr = thresholds["iqr"]
    mad_thr = thresholds["mad"]

    print("\n[INFO] Thresholds (v2)")
    print(f" - TOP {top_percent:.1f}% 기준 threshold : {top_thr:.4f}")
    print(
        f" - IQR 기반 threshold (Q3 + 1.5*IQR): {iqr_thr:.4f} "
        f"(Q1={thresholds['q1']:.4f}, Q3={thresholds['q3']:.4f}, IQR={thresholds['iqr_raw']:.4f})"
    )
    print(
        f" - MAD 기반 threshold (median + {mad_k:.1f} * MAD): {mad_thr:.4f} "
        f"(median={thresholds['median']:.4f}, MAD={thresholds['mad_raw']:.4f})"
    )

    # flag 컬럼 추가
    df["is_anomaly_topN"] = df["recon_loss"] >= top_thr
    df["is_anomaly_iqr"] = df["recon_loss"] >= iqr_thr
    df["is_anomaly_mad"] = df["recon_loss"] >= mad_thr

    print("\n[INFO] Anomaly counts (v2)")
    print(
        f" - TOP {top_percent:.1f}% 기준 이상치 수 : {df['is_anomaly_topN'].sum()} "
        f"/ {len(df)}"
    )
    print(
        f" - IQR 기준 이상치 수           : {df['is_anomaly_iqr'].sum()} "
        f"/ {len(df)}"
    )
    print(
        f" - MAD 기준 이상치 수           : {df['is_anomaly_mad'].sum()} "
        f"/ {len(df)}"
    )

    # ------------------------
    # 6) CSV 저장
    # ------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] anomaly scores (v2) → {save_path}")
    print("\n[SUCCESS] Anomaly scoring v2 complete.\n")


if __name__ == "__main__":
    anomaly_scoring_v2(device="cpu")
