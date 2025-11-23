# utils/dataset_v2.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


class NewsKOSPIFusionDatasetV2(Dataset):

    def __init__(
        self,
        csv_path: str,
        numeric_cols,
        emb_cache_path: str = "../data/processed/kobert_emb_v2.pt",
        scaler=None,
        device: str = "cpu",
    ):
        self.csv_path = csv_path
        self.numeric_cols = list(numeric_cols)
        self.emb_cache_path = emb_cache_path
        self.scaler = scaler
        self.device = device

        # -----------------------------
        # 1) CSV 로드
        # -----------------------------
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        df["날짜"] = pd.to_datetime(df["날짜"])
        df["뉴스텍스트"] = df["뉴스텍스트"].fillna("").astype(str)

        self.df = df
        self.texts = df["뉴스텍스트"].tolist()

        # -----------------------------
        # 2) 수치형 + 스케일링
        # -----------------------------
        num = df[self.numeric_cols].astype(float).values

        if self.scaler is not None:
            num = self.scaler.transform(num)

        self.numeric = num.astype(np.float32)

        # -----------------------------
        # 3) KoBERT 임베딩 처리
        # -----------------------------
        if os.path.exists(self.emb_cache_path):
            print("[DatasetV2] Loading KoBERT embeddings from cache...")
            emb = torch.load(self.emb_cache_path, map_location="cpu")

            if isinstance(emb, torch.Tensor) and emb.shape[0] == len(self.df):
                self.text_emb = emb.float()
            else:
                print("[WARN] Embeddings mismatch — rebuilding.")
                self._build_text_embeddings()
        else:
            self._build_text_embeddings()

        if isinstance(self.text_emb, torch.Tensor):
            self.text_emb = self.text_emb.float()


    def _build_text_embeddings(self):
        print("[DatasetV2] Encoding news with KoBERT...")

        model_name = "skt/kobert-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert = AutoModel.from_pretrained(model_name).to(self.device)
        bert.eval()

        all_embs = []

        with torch.no_grad():
            for text in self.texts:
                text = text.strip()
                if not text:
                    all_embs.append(torch.zeros(768))
                    continue

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = bert(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                all_embs.append(cls_emb.squeeze(0).cpu())

        all_embs = torch.stack(all_embs, dim=0)

        os.makedirs(os.path.dirname(self.emb_cache_path), exist_ok=True)
        torch.save(all_embs, self.emb_cache_path)

        print(f"[DatasetV2] Saved KoBERT embeddings → {self.emb_cache_path}")

        self.text_emb = all_embs


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        num_feat = torch.from_numpy(self.numeric[idx])
        txt_feat = self.text_emb[idx].float()

        features = torch.cat([num_feat, txt_feat], dim=-1)

        return {
            "features": features,  # 날짜 제거!
        }
