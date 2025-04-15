import requests


class LLMAnalyzer:
    def __init__(self, model_name="llama3.2"):
        self.model = model_name
        self.url = "http://localhost:11434/api/generate"

    def get_sentiment(self, news_text):
        prompt = f"""
        You are a crypto sentiment expert.
        Analyze the following news and reply with one word: Bullish, Bearish, or Neutral.

        News: "{news_text}"
        """
        payload = {"model": self.model, "prompt": prompt, "stream": False}

        try:
            res = requests.post(self.url, json=payload)
            res.raise_for_status()
            response_text = res.json()["response"].strip().capitalize()
            return response_text

        except Exception as e:
            print(f"❌ Error querying LLM: {e}")
            return "Error"


class CryptoNewsFetcher:
    def __init__(self, api_key, currencies="BTC,ETH"):
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        self.currencies = currencies

    def fetch_headlines(self, kind="news", filter="latest", limit=10):
        params = {
            "auth_token": self.api_key,
            "currencies": self.currencies,
            "filter": filter,
            "kind": kind,
            "public": "true",
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            news_items = []
            for item in data["results"][:limit]:
                title = item["title"]
                published_at = item["published_at"]  # Format: "2024-04-13T15:24:00Z"
                news_items.append({"title": title, "timestamp": published_at})

            return news_items

        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return []
