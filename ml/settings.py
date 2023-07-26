import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    current_api_key = os.getenv("CURRENT_API_KEY")
    g_news_api_key = os.getenv("GNEWS_API_KEY")
    news_api_key = os.getenv("NEWS_API_KEY")
    media_stack_api_key = os.getenv("MEDIA_STACK_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")


settings = Settings()
