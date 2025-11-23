# src/config.py
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(r"C:\MyProject\PytorchProject")

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_NEWS_DIR = DATA_DIR / "raw" / "news"
RAW_KOSPI_DIR = DATA_DIR / "raw" / "kospi"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
AE_MODEL_DIR = MODELS_DIR / "autoencoder"
EMBEDDING_DIR = MODELS_DIR / "embeddings"
TOPIC_DIR = MODELS_DIR / "topic"          # ⭐ 추가됨

# Project date range
START_DATE = date(2023, 11, 1)
END_DATE = date(2025, 10, 31)

# Create directories if missing
for d in [
    RAW_NEWS_DIR,
    RAW_KOSPI_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    AE_MODEL_DIR,
    EMBEDDING_DIR,
    TOPIC_DIR
]:
    d.mkdir(parents=True, exist_ok=True)
