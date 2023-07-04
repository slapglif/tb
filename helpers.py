import pandas as pd


def clear_trend(data: pd.DataFrame) -> bool:
    """
    Identifies if a clear trend is present in the market.

    Parameters:
    data (pd.DataFrame): Historical stock data.

    Returns:
    bool: True if there is a clear trend in the market, False otherwise.
    """
    # Calculate the moving averages
    short_ma = data['close'].rolling(window=20).mean()
    long_ma = data['close'].rolling(window=50).mean()

    # Determine if there is a clear trend
    if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] < long_ma.iloc[-2]:
        return True
    elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] > long_ma.iloc[-2]:
        return True
    else:
        return False



def avoid_peaks(api, stock):
    # Get historical data for the stock
    stock_data = api.get_barset(stock, 'day', limit=200).df[stock]

    # Calculate moving averages
    sma_5 = stock_data['close'].rolling(window=5).mean()
    sma_10 = stock_data['close'].rolling(window=10).mean()

    # Check if the current price is above the moving averages
    current_price = stock_data['close'][-1]
    if current_price > sma_5[-1] and current_price > sma_10[-1]:
        return True
    else:
        return False


