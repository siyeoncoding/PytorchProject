import pandas as pd
from pathlib import Path
import glob

# ★ 너의 절대 경로
RAW_DIR = Path(r"C:\MyProject\PytorchProject\data\raw\news")

# 병합 결과 저장 경로
MERGED_PATH = RAW_DIR / "naver_news_merged.parquet"

def merge_all_news():
    print("=== MERGE START ===")
    print("Loading directory:", RAW_DIR)

    # 모든 parquet 파일 리스트
    files = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    print(f"Found {len(files)} parquet files")

    dfs = []

    for f in files:
        print("Loading:", f)
        df = pd.read_parquet(f)
        dfs.append(df)

    # 병합
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Rows before drop_duplicates: {len(df_all)}")

    # 중복 제거
    df_all.drop_duplicates(subset=["pub_datetime", "title"], inplace=True)
    print(f"Rows after drop_duplicates: {len(df_all)}")

    # 정렬
    df_all.sort_values("pub_datetime", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # 저장
    df_all.to_parquet(MERGED_PATH, index=False)
    print("Saved merged file →", MERGED_PATH)

if __name__ == "__main__":
    merge_all_news()
