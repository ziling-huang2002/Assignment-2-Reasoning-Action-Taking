import os
from tavily import TavilyClient

# 1. 直接使用 Tavily API 客戶端

def search_google(query):
    """
    使用 Tavily API 進行搜尋並回傳結果。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in .env")

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=3)
    return response
