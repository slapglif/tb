from typing import List
import alpaca_trade_api as tradeapi

def scan_market(api: tradeapi.REST, stocks: List[str], min_volume: float, min_price: float, max_price: float) -> List[str]:
    """
    Scans the market and returns a list of symbols to trade based on filter criteria.

    Parameters:
    api (tradeapi.REST): Alpaca API instance
    stocks (List[str]): List of stocks to scan
    min_volume (float): Minimum volume threshold for filtering
    min_price (float): Minimum price threshold for filtering
    max_price (float): Maximum price threshold for filtering

    Returns:
    List[str]: List of filtered symbols
    """
    filtered_stocks = []
    for stock in stocks:
        barset = api.get_barset(stock, 'day', limit=2)
        if not barset or not barset[stock]:
            continue
        bars = barset[stock]
# Filter by volume, VWAP, and price
        if (bars[-1].v > min_volume and bars[-1].v > bars[-2].v
                and min_price <= bars[-1].vwap <= max_price and min_price <= bars[-1].c <= max_price):
            filtered_stocks.append(stock)

    return filtered_stocks
