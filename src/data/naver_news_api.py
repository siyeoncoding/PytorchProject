import os
import json
import time
import datetime
import email.utils
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
RAW_DIR = Path(r"C:\MyProject\PytorchProject\data\raw\news")
RAW_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

if not client_id or not client_secret:
    raise ValueError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 값을 .env에 설정해야 합니다.")

# ---------------------------------------------------------
# NAVER API
# ---------------------------------------------------------
def naver_news_search(query: str, display=100, start=1):
    enc_query = urllib.parse.quote(query)
    url = (
        f"https://openapi.naver.com/v1/search/news?"
        f"query={enc_query}&display={display}&start={start}&sort=date"
    )

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)

    response = urllib.request.urlopen(request)
    if response.getcode() != 200:
        print("Error Code:", response.getcode())
        return None

    return json.loads(response.read().decode("utf-8"))


def parse_pubdate(pubdate_str: str) -> datetime.datetime:
    """네이버 pubDate → datetime"""
    return email.utils.parsedate_to_datetime(pubdate_str)


# ---------------------------------------------------------
# 월별 뉴스 수집
# ---------------------------------------------------------
def fetch_news_for_query_month(query: str, year: int, month: int) -> pd.DataFrame:
    """
    특정 월(year-month) + 특정 키워드(query) 뉴스 1000건까지 수집
    """
    monthly_rows = []
    start = 1
    max_start = 1000
    display = 100

    # 월별 쿼리 생성
    q = f"{query} {year}년 {month:02d}월"

    while True:
        data = naver_news_search(q, display=display, start=start)
        if not data:
            break

        items = data.get("items", [])
        if not items:
            break

        for it in items:
            pub_dt = parse_pubdate(it["pubDate"])
            monthly_rows.append({
                "pub_datetime": pub_dt,
                "pub_date": pub_dt.date(),
                "query": query,
                "query_month": f"{year}-{month:02d}",
                "title": it.get("title"),
                "description": it.get("description"),
                "link": it.get("link"),
                "originallink": it.get("originallink"),
            })

        start += display
        if start > max_start:
            break

        time.sleep(0.2)

    df = pd.DataFrame(monthly_rows)
    if not df.empty:
        df = df.sort_values("pub_datetime").reset_index(drop=True)

    return df


# ---------------------------------------------------------
# 월 리스트 생성
# ---------------------------------------------------------
def generate_month_list(start_year: int, start_month: int, end_year: int, end_month: int):
    """
    2023-11 ~ 2025-10 같은 월 리스트 생성
    """
    months = []

    cur = datetime.date(start_year, start_month, 1)
    end = datetime.date(end_year, end_month, 1)

    while cur <= end:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = datetime.date(cur.year + 1, 1, 1)
        else:
            cur = datetime.date(cur.year, cur.month + 1, 1)

    return months


# ---------------------------------------------------------
# 전체 수집 실행
# ---------------------------------------------------------
def fetch_news_fixed_range():

    # 🔥 고정된 날짜 범위 (너가 원하는 기준)
    start_year, start_month = 2023, 11
    end_year, end_month = 2025, 10

    # 월 리스트 생성
    months = generate_month_list(start_year, start_month, end_year, end_month)

    # 키워드 목록
    queries = ["한국 증시", "코스피", "코스닥"]

    all_dfs = []

    for query in queries:
        for (y, m) in months:
            print(f"[FETCH] {query} / {y}-{m:02d}")

            df = fetch_news_for_query_month(query, y, m)

            if df.empty:
                print(f"    → No data for {query} {y}-{m:02d}")
                continue

            # 월별 저장
            filename = f"news_{query.replace(' ', '_')}_{y}_{m:02d}.parquet"
            path = RAW_DIR / filename
            df.to_parquet(path, index=False)

            print(f"    → Saved {len(df)} rows → {path}")

            all_dfs.append(df)

    # 전체 병합
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)

        df_all = df_all.drop_duplicates(subset=["pub_datetime", "title"])
        df_all = df_all.sort_values("pub_datetime").reset_index(drop=True)

        merged_path = RAW_DIR / "naver_news_2023-11_2025-10.parquet"
        df_all.to_parquet(merged_path, index=False)

        print("\n==============================")
        print(f"[SAVED] Total merged news: {len(df_all)} rows")
        print(f"[PATH] {merged_path}")
        print("==============================")

        return merged_path

    print("No news collected.")
    return None


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    fetch_news_fixed_range()
