from typing import List, Optional
import time
import threading

from typing import Dict, List
import pandas as pd
import alpaca_trade_api as tradeapi


import alpaca_trade_api as tradeapi

from rules import Trade, TradingLogic

from typing import List
import time

import alpaca_trade_api as tradeapi

from actors import Actor
from models import Trade, Position
from rules import RuleEngine


class TradeActor(Actor):
    def __init__(self, api_key: str, api_secret: str, base_url: str, symbol: str, rule_engine: RuleEngine):
        super().__init__()
        self.api = tradeapi.REST(api_key, api_secret, base_url=base_url)
        self.symbol = symbol
        self.rule_engine = rule_engine

    def perform_action(self):
        # Fetch stock data from Alpaca API
        barset = self.api.get_barset(self.symbol, 'minute', limit=1)
        stock_data = barset[self.symbol][0]

        # Calculate trade signal based on rules
        trade_signal = self.rule_engine.calculate_signal(stock_data)

        # Check if there is an open position
        position = self.api.get_position(self.symbol)
        if position:
            position_qty = int(position.qty)
            if trade_signal == 'sell':
                # Close the position if the signal is to sell
                self.api.submit_order(self.symbol, position_qty, 'sell', 'market', 'day')
                self.log_trade(position_qty, 'sell', stock_data)
            elif self.rule_engine.should_reverse_position(position, stock_data):
                # Reverse the position if the rule engine says to do so
                self.reverse_position(position_qty, position, stock_data)
            elif self.rule_engine.should_close_position(position, stock_data):
                # Close the position if the rule engine says to do so
                self.close_position(position_qty, position, stock_data)
        else:
            if trade_signal == 'buy':
                # Calculate the quantity to buy based on risk management rules
                qty = self.rule_engine.calculate_quantity(self.api, stock_data)

                # Submit a buy order if the signal is to buy
                self.api.submit_order(self.symbol, qty, 'buy', 'limit', 'day', stock_data.close,
                                       self.rule_engine.calculate_stop_loss(stock_data),
                                       self.rule_engine.calculate_take_profit(stock_data))
                self.log_trade(qty, 'buy', stock_data)

    def log_trade(self, qty: int, action: str, stock_data):
        # Create a Trade object and log it to the database
        trade = Trade(self.symbol, qty, action, stock_data.time, stock_data.close)
        self.log_event(trade.to_dict())

    def reverse_position(self, position_qty: int, position: Position, stock_data):
        # Close the current position
        self.api.submit_order(self.symbol, position_qty, 'sell', 'market', 'day')
        self.log_trade(position_qty, 'sell', stock_data)

        # Wait for the order to fill
        time.sleep(2)

        # Calculate the new quantity to buy based on risk management rules
        qty = self.rule_engine.calculate_quantity(self.api, stock_data)

        # Submit a buy order to reverse the position
        self.api.submit_order(self.symbol, qty, 'buy', 'limit', 'day', stock_data.close,
                               TradingLogic.calculate_stop_loss(stock_data),
                               self.rule_engine.calculate_take_profit(stock_data))
        self.log_trade(qty, 'buy', stock_data)

    def close_position(self, qty: int) -> Optional[Trade]:
        """Submit a sell order to close the position"""
        if self.position == 0:
            return None

        # Determine the quantity to sell
        qty = min(qty, abs(self.position))

        # Submit a sell order to close the position
        self.api.submit_order(
            symbol=self.symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day'
        )

        # Log the trade
        trade = self.log_trade(qty, TradeType.SELL)

        # Update the position and cash balance
        self.position -= qty
        self.cash += trade.proceeds

        return trade



class StockDataActor:
    """
    Code Analysis

    Main functionalities:
    The StockDataActor class is designed to retrieve and store real-time stock data for a given symbol and timeframe using the Alpaca Trade API. It allows users to get the latest bar for a symbol and update the current data with the latest bar. The class returns a pandas DataFrame containing the current data.

    Methods:
    - __init__(self, api: tradeapi.REST, symbol: str, timeframe: str): Initializes the class with the Alpaca Trade API, symbol, and timeframe.
    - get_data(self) -> pd.DataFrame: Retrieves the latest bar for the symbol and updates the current data with the latest bar. Returns a pandas DataFrame containing the current data.

    Fields:
    - api: An instance of the Alpaca Trade API.
    - symbol: A string representing the stock symbol.
    - timeframe: A string representing the timeframe for the stock data.
    - current_data: A pandas DataFrame containing the current stock data.
    """
    def __init__(self, api: tradeapi.REST, symbol: str, timeframe: str):
        self.api = api
        self.symbol = symbol
        self.timeframe = timeframe
        self.current_data = None

    def get_data(self) -> pd.DataFrame:
        # Get the latest bar for the symbol
        latest_bar = self.api.get_barset(self.symbol, self.timeframe, limit=1)[self.symbol][0]
        # Update the current data with the latest bar
        if self.current_data is None:
            self.current_data = pd.DataFrame([latest_bar._raw])
        else:
            self.current_data = self.current_data.append([latest_bar._raw], ignore_index=True)
        return self.current_data

class RuleActor:
    def __init__(self, api: tradeapi.REST, stock_data_actor: StockDataActor, rules: Dict[str, List]):
        self.api = api
        self.stock_data_actor = stock_data_actor
        self.rules = rules

    def check_rules(self) -> bool:
        data = self.stock_data_actor.get_data()
        if data.empty:
            return False

        # Check entry rules
        if 'entry' in self.rules:
            entry_rules = self.rules['entry']
            for rule in entry_rules:
                if not rule.check(data):
                    return False

        # Check exit rules
        if 'exit' in self.rules:
            exit_rules = self.rules['exit']
            for rule in exit_rules:
                if rule.check(data):
                    return False

        # Check reverse rules
        if 'reverse' in self.rules:
            reverse_rules = self.rules['reverse']
            for rule in reverse_rules:
                if rule.check(data):
                    return True

        # No rules triggered
        return False

