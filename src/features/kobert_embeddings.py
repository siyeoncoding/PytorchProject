# src/features/kobert_embeddings.py
from pathlib import Path

import device
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src.config import INTERIM_DIR, EMBEDDING_DIR


# 감성 모델 로드
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sent_tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
sent_model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base").to(device)
sent_model.eval()


KOBERT_MODEL = "skt/kobert-base-v1"
MAX_LEN = 256
TS_PATH = INTERIM_DIR / "merged_timeseries.parquet"


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def compute_sentiment(text_list):
    inputs = sent_tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = sent_model(**inputs).logits.cpu()
    prob = torch.softmax(output, dim=1)
    score = prob[:, 1] - prob[:, 0]  # positive - negative
    return score.unsqueeze(1)


def build_embeddings(device="cpu"):
    df = pd.read_parquet(TS_PATH)
    texts = df["merged_text"].fillna("").tolist()

    tokenizer = AutoTokenizer.from_pretrained(KOBERT_MODEL)
    model = AutoModel.from_pretrained(KOBERT_MODEL).to(device)
    model.eval()

    ds = TextDataset(texts)
    dl = DataLoader(ds, batch_size=16, shuffle=False)

    all_emb = []
    all_sent = []
    all_topic = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="KoBERT Embedding"):
            enc = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            )

            # 🔥 핵심 패치: token_type_ids = 0 ONLY
            enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].cpu()

            all_emb.append(cls_emb)

            # 감성/토픽 더미(0)
            s = torch.zeros(cls_emb.size(0), 1)
            t = torch.zeros(cls_emb.size(0), 1)

            all_sent.append(s)
            all_topic.append(t)

    emb = torch.cat(all_emb)
    sent = torch.cat(all_sent)
    topic = torch.cat(all_topic)

    out_path = EMBEDDING_DIR / "kobert_embeddings.pt"
    torch.save(
        {
            "embeddings": emb,
            "sentiments": sent,
            "topics": topic
        },
        out_path
    )

    print("[KOBERT EMBEDDING SAVED]")
    print(" →", out_path)
    print(" → shape:", emb.shape)


if __name__ == "__main__":
    build_embeddings(device="cpu")
