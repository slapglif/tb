import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    alpaca_key = os.getenv("ALPACA_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET")
    alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    alpaca_paper = os.getenv("ALPACA_PAPER", 'https://paper-api.alpaca.markets')
    alpaca_live = os.getenv("ALPACA_LIVE")
    alpaca_live_key = os.getenv("ALPACA_LIVE_KEY")
    alpaca_live_secret = os.getenv("ALPACA_LIVE_SECRET")
    alpaca_live_base_url = os.getenv("ALPACA_LIVE_BASE_URL")