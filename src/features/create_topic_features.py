# src/features/create_topic_features.py
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from src.config import INTERIM_DIR, TOPIC_DIR

def build_topics():
    # 🔥 SENTIMENT 포함된 파일로 변경
    df = pd.read_parquet(INTERIM_DIR / "timeseries_with_sentiment.parquet")
    texts = df["merged_text"].fillna("").tolist()

    embed_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    emb = embed_model.encode(texts, show_progress_bar=True)

    topic_model = BERTopic(language="multilingual", calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(texts, emb)

    topic_model.save(TOPIC_DIR / "bertopic_model")

    df["topic_id"] = topics

    t_dim = max(topics) + 1
    topic_vec = np.zeros((len(topics), t_dim))
    for i, t in enumerate(topics):
        topic_vec[i, t] = 1

    df_topics = pd.DataFrame(topic_vec, columns=[f"topic_{i}" for i in range(t_dim)])
    df_final = pd.concat([df, df_topics], axis=1)

    out = INTERIM_DIR / "timeseries_with_sentiment_topic.parquet"
    df_final.to_parquet(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    build_topics()
