# src/features/create_sentiment_features.py
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from src.config import INTERIM_DIR

# ❗ 감성분석은 BEOMI/kcbert-base 아닌, 실제 감성 fine-tuned 모델을 사용해야 더 정확함
MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # 예시 모델 (한국어 가능)
# 네가 원하면 한국어 금융 특화 감성 모델로 교체해줄게

TS_PATH = INTERIM_DIR / "merged_timeseries.parquet"


def build_sentiment(device="cpu"):
    df = pd.read_parquet(TS_PATH)
    texts = df["merged_text"].fillna("").tolist()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval()

    scores = []

    print(f"[INFO] Total texts = {len(texts)}")
    print("[INFO] Running sentiment analysis...")

    for t in tqdm(texts, desc="Sentiment scoring"):
        enc = tok(t, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)

        with torch.no_grad():
            logits = model(**enc).logits.cpu()

        prob = torch.softmax(logits, dim=1)

        # 감성 점수 계산 방식 (모델마다 달라짐 → 여기서는 예시)
        # 1~5 star → -1 ~ +1으로 변환
        cls_idx = prob.argmax().item()
        score = (cls_idx - 2) / 2  # 0~4 → -1 ~ +1로 매핑

        scores.append(score)

    df["sentiment"] = scores

    out = INTERIM_DIR / "timeseries_with_sentiment.parquet"
    df.to_parquet(out, index=False)

    print("\n[SAVED] sentiment features →", out)


if __name__ == "__main__":
    build_sentiment(device="cpu")
