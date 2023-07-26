import json
import sys

from requests import Session
from settings import settings
from datetime import datetime, timedelta
import openai
from loguru import logger

openai.api_key = settings.openai_api_key

logger.configure(
    handlers=[
        dict(
            sink="logs/news.log",
            level="DEBUG",
            diagnose=False,
            backtrace=False,
            format='{message}',
        ),
        # dict(
        #     sink=sys.stdout,
        #     level="DEBUG",
        #     diagnose=True,
        #     backtrace=True,
        # ),
    ]
)


def get_date(weeks):
    now = datetime.now()
    two_weeks_ago = now - timedelta(weeks=weeks)
    return two_weeks_ago.strftime("%Y-%m-%dT%H:%M:%S.%f")


class DataGathering:
    def __init__(self):
        self.session = Session()
        self.session.headers.update({"Authorization": settings.current_api_key})

    def current_api(self, symbol):
        url = "https://api.currentsapi.services/v1/search"
        params = {
            "start_date": get_date(2),
            "end_date": get_date(0),
            "keywords": symbol,
            "limit": 20,
        }
        response = self.session.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data from current-api: {response.text}")
        return response.json()

    def g_news(self, symbol):
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": symbol,
            "apikey": settings.g_news_api_key,
        }
        response = self.session.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data from g-news-api: {response.text}")
        return response.json()

    def news_api(self, symbol):
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "sortBy": "popularity",
            "apiKey": settings.news_api_key,
        }
        response = self.session.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data from news-api: {response.text}")
        return response.json()

    def media_stack(self, symbol):
        url = "http://api.mediastack.com/v1/news"
        params = {
            "access_key": settings.media_stack_api_key,
            "keywords": symbol,
            "limit": 20,
        }
        response = self.session.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data from stack-media: {response.text}")
        return response.json()


def query_ai(symbol, news):
    return openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "system",
                "content": "Want assistance provided by qualified individuals enabled with experience on understanding charts using technical analysis tools while interpreting macroeconomic environment prevailing across world consequently assisting customers acquire long term advantages requires clear verdicts therefore seeking same through informed predictions written down precisely! First statement contains following content",
            },
            {
                "role": "user",
                "content": f"review the news data using sentiment analysis, and build your knowledge base on it, do not reply anything until I ask for. here is the news: {json.dumps(news)}",
            },
        ],
        temperature=0.2,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


if __name__ == "__main__":
    data_gathering = DataGathering()
    symbol = input("\nenter your asset... ")
    gather_news = {
        "current_api": data_gathering.current_api(symbol),
        "g_news": data_gathering.g_news(symbol),
        "news_api": data_gathering.news_api(symbol),
        "media_stack": data_gathering.media_stack(symbol),
    }
    logger.info(gather_news)
#     for value in gather_news.values():
#         query_ai(symbol, value)
#
#
# def query_ai(symbol, news):
#     return openai.ChatCompletion.create(
#         model="gpt-4-0613",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Want assistance provided by qualified individuals enabled with experience on understanding charts using technical analysis tools while interpreting macroeconomic environment prevailing across world consequently assisting customers acquire long term advantages requires clear verdicts therefore seeking same through informed predictions written down precisely! First statement contains following content",
#             },
#             {
#                 "role": "user",
#                 "content": f"review the news data using sentiment analysis, and build your knowledge base on it, do not reply anything until I ask for. here is the news: {json.dumps(news)}",
#             },
#         ],
#         temperature=0.2,
#         max_tokens=64,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#     )
