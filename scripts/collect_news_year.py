import urllib.request
import urllib.parse
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import os

client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")

# -------------------------
# 네이버 뉴스 검색 API 요청 함수
# -------------------------
def naver_news_search(query="한국 증시", display=100, start=1):
    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news?query={enc_query}&display={display}&start={start}"

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)

    response = urllib.request.urlopen(request)
    rescode = response.getcode()

    if rescode != 200:
        print("Error Code:", rescode)
        return None

    response_body = response.read().decode("utf-8")
    return json.loads(response_body)


# -------------------------
# 1년 동안 모든 날짜 뉴스 반복 수집
# -------------------------
def collect_news_one_year(query="한국 증시", days=365):
    news_items = []
    today = datetime.now()
    start_date = today - timedelta(days=days)

    print(f"[INFO] 1년치 뉴스 수집 시작: {start_date.date()} ~ {today.date()}")

    # 하루 단위로 반복
    date_cursor = start_date
    while date_cursor <= today:
        date_str = date_cursor.strftime("%Y-%m-%d")
        print(f"\n[INFO] 날짜 처리 중: {date_str}")

        # 100개 단위 * 3페이지 = 300개
        for p in range(3):
            start_idx = p * 100 + 1
            data = naver_news_search(query=query, display=100, start=start_idx)

            if not data or "items" not in data:
                continue

            for item in data["items"]:
                news_items.append({
                    "date": date_str,
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "link": item.get("originallink") or item.get("link"),
                    "pubDate": item.get("pubDate")
                })

            time.sleep(0.2)

        date_cursor += timedelta(days=1)

    return pd.DataFrame(news_items)


if __name__ == "__main__":
    df = collect_news_one_year("한국 증시", days=365)
    df.to_csv("../data/raw/news_raw.csv", encoding="utf-8-sig", index=False)
    print("[SUCCESS] 1년치 네이버 뉴스 저장 완료 → data/raw/news_rawtest.csv")
