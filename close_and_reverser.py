import alpaca_trade_api
from alpaca_trade_api.entity import Position

def close_and_reverse(api: alpaca_trade_api.REST, position: Position) -> bool:
    """
    Close the current position and reverse it to the opposite direction (from long to short or vice versa).

    Returns True if the position was closed and reversed successfully, False otherwise.
    """
    # Get the current position
    symbol = position.symbol
    qty = abs(int(float(position.qty)))
    side = 'sell' if position.side == 'long' else 'buy'

    # Close the current position
    try:
        api.close_position(symbol)
    except Exception as e:
        print(f"Error closing position: {e}")
        return False

    # Reverse the position
    try:
        # Submit a new order with opposite side
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            stop_loss={'stop_price': position.avg_entry_price * (1 + 0.02),
                       'limit_price': position.avg_entry_price * (1 + 0.03)},
            take_profit={'limit_price': position.avg_entry_price * (1 - 0.01)}
        )
    except Exception as e:
        print(f"Error reversing position: {e}")
        return False

    return True
