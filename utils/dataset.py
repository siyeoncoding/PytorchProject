# utils/dataset.py
import os
from typing import Sequence, Dict, Any, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


class NewsStockDataset(Dataset):
    """
    KOSPI + 뉴스 텍스트 멀티모달 데이터셋

    - csv_path: ../data/processed/merged_kospi_news.csv
    - numeric_cols: 수치 피처 컬럼 리스트 (예: 종가, return, volatility)
    - text_col: 뉴스 텍스트 컬럼명 (예: '뉴스텍스트')

    특징
    ----
    - 뉴스텍스트 NaN 자동 처리 (빈 문자열로 바꾸고 KoBERT 인코딩)
    - KoBERT 임베딩 캐시 지원 (emb_save_path에 저장/재사용)
    """

    def __init__(
        self,
        csv_path: str = "../data/processed/merged_kospi_news.csv",
        numeric_cols: Sequence[str] = ("종가", "return", "volatility"),
        text_col: str = "뉴스텍스트",
        max_length: int = 256,
        device: str = "cpu",
        use_cached_emb: bool = True,
        emb_save_path: str = "../data/processed/kobert_emb.pt",
    ):
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

        # -----------------------
        # 1) 데이터 로드
        # -----------------------
        self.df = pd.read_csv(csv_path)
        self.numeric_cols = list(numeric_cols)
        self.text_col = text_col
        self.max_length = max_length
        self.device = device
        self.use_cached_emb = use_cached_emb
        self.emb_save_path = emb_save_path

        # 날짜 컬럼 이름 저장 (있으면)
        self.date_col = "날짜" if "날짜" in self.df.columns else None

        # 뉴스 NaN 자동 처리 → 빈 문자열
        if self.text_col in self.df.columns:
            na_cnt = self.df[self.text_col].isna().sum()
            print(f"[INFO] 뉴스텍스트 NaN 개수: {na_cnt}")
            self.df[self.text_col] = self.df[self.text_col].fillna("")

        # 수치형 피처 텐서
        self.numeric_tensor = torch.tensor(
            self.df[self.numeric_cols].values,
            dtype=torch.float32,
        )

        print(f"[INFO] 전체 샘플 수: {len(self.df)}")
        print(f"[INFO] numeric_cols: {self.numeric_cols}")

        # -----------------------
        # 2) KoBERT 로드
        # -----------------------
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
        self.model = AutoModel.from_pretrained("skt/kobert-base-v1")
        self.model.to(self.device)
        self.model.eval()

        # -----------------------
        # 3) KoBERT 임베딩 캐시 로직
        # -----------------------
        self.text_emb_tensor = None

        if self.use_cached_emb and os.path.exists(self.emb_save_path):
            cached = torch.load(self.emb_save_path, map_location="cpu")
            if cached.shape[0] == len(self.df):
                print(f"[INFO] 캐시된 KoBERT 임베딩 로드: {self.emb_save_path}")
                self.text_emb_tensor = cached
            else:
                print(
                    f"[WARN] 캐시 행 수({cached.shape[0]}) != 데이터 행 수({len(self.df)}) → 재계산"
                )
                self._build_and_cache_embeddings()
        elif self.use_cached_emb:
            print("[INFO] 캐시된 임베딩이 없어 새로 계산합니다.")
            self._build_and_cache_embeddings()
        # use_cached_emb=False 이면 __getitem__에서 on-the-fly 인코딩

    # ==========================================
    # 내부: 한 문장 → KoBERT 임베딩
    # ==========================================
    def _encode_text(self, text: str) -> torch.Tensor:
        # NaN / None 안전 처리
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            # KoBERT에서 token_type_ids 제거 필요
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
            return cls_emb.squeeze(0).cpu()  # (hidden_size,)

    # ==========================================
    # 내부: 전체 텍스트 임베딩 미리 계산 + 캐시
    # ==========================================
    def _build_and_cache_embeddings(self):
        print("[INFO] KoBERT 임베딩 전체 계산 중...")
        texts: List[str] = self.df[self.text_col].tolist()
        all_embs: List[torch.Tensor] = []

        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0 or i == 0:
                print(f" - {i + 1}/{len(texts)}개 처리 중")
            emb = self._encode_text(text)
            all_embs.append(emb)

        self.text_emb_tensor = torch.stack(all_embs, dim=0)  # (N, hidden_size)

        os.makedirs(os.path.dirname(self.emb_save_path), exist_ok=True)
        torch.save(self.text_emb_tensor, self.emb_save_path)
        print(f"[SUCCESS] KoBERT 임베딩 저장 완료 → {self.emb_save_path}")

    # ==========================================
    # 필수 메서드
    # ==========================================
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 수치형 피처
        numeric_feat = self.numeric_tensor[idx]  # (num_numeric,)

        # 텍스트 임베딩 (캐시 있으면 사용, 없으면 즉석 계산)
        if self.text_emb_tensor is not None:
            text_emb = self.text_emb_tensor[idx]
        else:
            raw_text = self.df.iloc[idx][self.text_col]
            text_emb = self._encode_text(raw_text)

        # 통합 피처
        all_features = torch.cat([numeric_feat, text_emb], dim=-1)

        sample: Dict[str, Any] = {
            "numeric": numeric_feat,   # (num_numeric,)
            "text_emb": text_emb,      # (hidden_dim,)
            "features": all_features,  # (num_numeric + hidden_dim,)
        }

        if self.date_col is not None:
            sample["date"] = self.df.iloc[idx][self.date_col]
        else:
            sample["date"] = None

        return sample


# ==========================================
# 디버그 전용 실행
# ==========================================
if __name__ == "__main__":
    ds = NewsStockDataset(
        csv_path="../data/processed/merged_kospi_news.csv",
        numeric_cols=("종가", "return", "volatility"),
        text_col="뉴스텍스트",
        use_cached_emb=False,  # 처음엔 False로 한두 개 확인해도 됨
        device="cpu",
    )

    print(f"[INFO] 전체 샘플 수: {len(ds)}")
    s = ds[0]
    print("[DEBUG] numeric:", s["numeric"].shape)
    print("[DEBUG] text_emb:", s["text_emb"].shape)
    print("[DEBUG] features:", s["features"].shape)
    print("[DEBUG] date:", s["date"])
