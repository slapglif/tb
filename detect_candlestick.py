from typing import List

import pandas as pd


def detect_candlestick_wick(candlestick_data: pd.DataFrame, wick_pct_threshold: float, ma_period: int) -> List[str]:
    """
    Analyzes the wick size of a candlestick to identify potential trade opportunities.

    Args:
        candlestick_data (pd.DataFrame): Historical candlestick data for a stock.
        wick_pct_threshold (float): The percentage threshold for the wick size relative to the candlestick body size.
        ma_period (int): The period of the moving average to use as a reference for determining the trend.

    Returns:
        List of stock symbols for which a potential trade opportunity has been detected based on the wick size.
    """
    trades = []
    for symbol in candlestick_data.symbol.unique():
        symbol_data = candlestick_data[candlestick_data.symbol == symbol]
        ma = symbol_data.close.rolling(window=ma_period).mean()

        for i in range(len(symbol_data)):
            open_price = symbol_data.open[i]
            close_price = symbol_data.close[i]
            high_price = symbol_data.high[i]
            low_price = symbol_data.low[i]
            body_size = abs(open_price - close_price)
            candle_range = high_price - low_price
            if open_price > close_price:
                upper_wick_pct = (high_price - open_price) / candle_range
                lower_wick_pct = (close_price - low_price) / candle_range
            else:
                upper_wick_pct = (high_price - close_price) / candle_range
                lower_wick_pct = (open_price - low_price) / candle_range

            # Check if the upper wick is large enough and below the MA
            if upper_wick_pct >= wick_pct_threshold and high_price < ma[i]:
                trades.append(symbol)
                break

            # Check if the lower wick is large enough and above the MA
            if lower_wick_pct >= wick_pct_threshold and low_price > ma[i]:
                trades.append(symbol)
                break

    return trades
