from typing import List, Dict
from alpaca_trade_api.rest import APIError

from close_and_reverser import close_and_reverse
from rules import calculate_quantity
from .entry_rules import *
from .exit_rules import *
from .reverse_rules import *
from .data import *
from .utils import *

import param


# Example function to submit an order
def submit_order(api, symbol, qty, side, order_type, time_in_force, limit_price=None, stop_price=None):
    if order_type == 'limit':
        if side == 'buy':
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='limit',
                time_in_force=time_in_force,
                limit_price=limit_price
            )
        elif side == 'sell':
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='limit',
                time_in_force=time_in_force,
                limit_price=limit_price
            )
    elif order_type == 'stop':
        if side == 'buy':
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='stop',
                time_in_force=time_in_force,
                stop_price=stop_price
            )
        elif side == 'sell':
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='stop',
                time_in_force=time_in_force,
                stop_price=stop_price
            )

    elif order_type == 'market':
        if side == 'buy':

            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force=time_in_force
            )
        elif side == 'sell':

            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force=time_in_force
            )


def apply_entry_rule(rule, stock_data):
    pass


def get_stock_data(api, stock):
    pass


def apply_exit_rule(rule, stock_data):
    pass


def apply_reverse_rule(rule, stock_data):
    pass


def amount_to_trade(api, quantity, param):
    amount = api.get_account().equity * param
    for x in param:

        amount = amount / x
        return min(amount, quantity)



def trade_stocks(api: AlpacaAPI, filtered_stocks: List[str], entry_rules: List[Dict], exit_rules: List[Dict], reverse_rules: List[Dict]):
    """
    Executes trades based on the logic defined in the other functions and customizable rules.
    """

    for stock in filtered_stocks:
        # Retrieve data for the stock
        stock_data = get_stock_data(api, stock)

        # Check if there are enough bars to support the rules
        if len(stock_data) < max(len(entry_rules), len(exit_rules), len(reverse_rules)):
            continue

        # Apply entry rules to generate signals
        signals = []
        for rule in entry_rules:
            if signal := apply_entry_rule(rule, stock_data):
                signals.append(signal)

        # Apply exit rules to generate signals
        for rule in exit_rules:
            if signal := apply_exit_rule(rule, stock_data):
                signals.append(signal)

        # Apply reverse rules to generate signals
        for rule in reverse_rules:
            if signal := apply_reverse_rule(rule, stock_data):
                signals.append(signal)

        # Remove duplicate signals
        signals = list(set(signals))

        # Execute trades based on signals
        for signal in signals:
            try:
                position = api.get_position(stock)
            except APIError:
                position = None

            # Close position and reverse it
            if 'close_and_reverse' in signal:
                if position:
                    close_and_reverse(api, position)

            # Open a new position
            elif 'entry' in signal:
                if not position:
                    quantity = calculate_quantity(api, stock, signal['stop_loss']['stop_price'], signal['risk'])
                    amount = amount_to_trade(api, quantity, signal['entry_price'])
                    try:
                        order = api.submit_order(
                            symbol=stock,
                            qty=quantity,
                            side=signal['side'],
                            type='limit',
                            time_in_force='gtc',
                            order_class='bracket',
                            stop_loss={'stop_price': signal['stop_loss']['stop_price']},
                            take_profit={'limit_price': signal['take_profit']['limit_price']},
                            limit_price=signal['entry_price'],
                            notional=amount
                        )
                        print(f"New position opened for {stock}: {signal['side']} {quantity} shares")
                    except APIError as e:
                        print(f"Failed to open new position for {stock}: {e}")

            # Close an existing position
            elif 'exit' in signal:
                if position:
                    try:
                        api.submit_order(
                            symbol=stock,
                            qty=position.qty,
                            side=signal['side'],
                            type='limit',
                            time_in_force='gtc',
                            order_class='bracket',
                            stop_loss={'stop_price': signal['stop_loss']['stop_price']},
                            take_profit={'limit_price': signal['take_profit']['limit_price']},
                            limit_price=signal['limit_price']
                        )
                        print(f"Position closed for {stock}: {signal['side']} {position.qty} shares")
                    except APIError as e:
                        print(f"Failed to close position for {stock}: {e}")
