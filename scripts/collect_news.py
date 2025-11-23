import urllib.request
import urllib.parse
import json
import pandas as pd
import time

client_id = ""
client_secret = ""

def naver_news_search(query="한국 증시", display=100, start=1):
    """
    네이버 뉴스 검색 API (JSON)
    query: 검색어
    display: 한 번에 가져오는 기사 수 (max=100)
    start: 시작점
    """
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

    response_body = response.read().decode('utf-8')
    result = json.loads(response_body)
    return result


def collect_news(query="한국 증시", pages=3):
    news_items = []

    for p in range(1, pages + 1):
        print(f"[INFO] API 요청 중: page {p}")

        data = naver_news_search(query=query, display=100, start=p * 100)
        if not data or "items" not in data:
            continue

        for item in data["items"]:
            news_items.append({
                "title": item.get("title"),
                "link": item.get("originallink") or item.get("link"),
                "description": item.get("description"),
                "pubDate": item.get("pubDate")
            })

        time.sleep(0.5)

    return pd.DataFrame(news_items)


if __name__ == "__main__":
    df = collect_news("한국 증시", pages=3)
    df.to_csv("../data/raw/news_raw.csv", encoding="utf-8-sig", index=False)
    print("[SUCCESS] 네이버 뉴스 API 저장 완료 → data/raw/news_raw.csv")
