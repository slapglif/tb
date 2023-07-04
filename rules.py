from typing import List

from alpaca_trade_api import entity

from close_and_reverser import close_and_reverse

from pydantic import BaseModel

from models import StockData
from typing import List


class Trade(BaseModel):
    symbol: str
    side: str
    qty: int
    price: float
    time: str


# Example function to close a position
def close_position(api, position):
    """
    The close_position function takes in an API object and a position object.
    It then closes the position by calling the close_position method on the API object,
    passing in the symbol of that position.

    Args:
        api: Access the api
        position: Pass the position object to the function

    Returns:
        A position object

    """
    api.close_position(position.symbol)


def calculate_quantity(api, stock_price, max_trade_value):
    """
    The calculate_quantity function takes in the API, stock price, and max trade value.
    It then gets the account information from Alpaca and calculates how many shares of a given stock can be bought with that buying power.
    The function returns an integer representing the number of shares that can be purchased.

    Args:
        api: Get the account information
        stock_price: Calculate the maximum number of shares that can be purchased
        max_trade_value: Limit the amount of money spent on a single trade

    Returns:
        The minimum of the number of shares that can be purchased with buying power and the max trade value

    """
    account = api.get_account()
    buying_power = float(account.buying_power)
    max_qty = int(buying_power / stock_price)
    return min(max_qty, int(max_trade_value / stock_price))


from typing import Tuple
import numpy as np


class TradeType:
    """
    The TradeType class is used to represent the type of trade.
    """
    BUY = 'buy'
    SELL = 'sell'


class TradingRule:
    """
    The TradingRule class is used to represent a trading rule.
    """

    def __init__(self, name, params):
        self.name = name
        self.params = params


class Signal:
    """
    The Signal class is used to represent a trading signal.
    """

    def __init__(self, action: str, quantity: float, stop_loss: float, take_profit: float):
        self.action = action
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit


class TradingLogic:
    """
    The TradingLogic class is used to represent the trading logic.
    """

    def __init__(self, position_size: float, stop_loss_pct: float, take_profit_pct: float):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the initial state of an object, and defines what information needs to be provided when creating one.

        Args:
            self: Represent the instance of the class
            position_size: float: Set the size of the position
            stop_loss_pct: float: Define the percentage of the position size that will be used as a stop loss
            take_profit_pct: float: Define the take profit percentage

        Returns:
            Nothing

        """
        self.current_price = 0
        self.previous_price = 0
        self.current_signal = Signal(action="", quantity=0, stop_loss=0, take_profit=0)
        self.previous_signal = Signal(action="", quantity=0, stop_loss=0, take_profit=0)
        self.previous_signal_time = None
        self.current_signal_time = None
        self.current_position = 0
        self.current_position_value = 0
        self.current_position_pct = 0
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def calculate_signal(self, buy_condition: bool, sell_condition: bool, current_position: float) -> Signal:
        """
        The calculate_signal function is the main function of this strategy. It takes in a boolean buy_condition,
        a boolean sell_condition, and a float current_position. The function returns either None or an instance of Signal
        depending on whether the conditions are met for buying or selling.

        Args:
            self: Refer to the object itself
            buy_condition: bool: Check if the buy condition is met
            sell_condition: bool: Determine if the sell condition is met
            current_position: float: Determine if the current position is a buy or sell

        Returns:
            A signal object

        """
        if buy_condition:
            quantity = self.calculate_quantity()
            stop_loss = self.calculate_stop_loss(buy=True)
            take_profit = self.calculate_take_profit(buy=True)
            return Signal(action=TradeType.BUY, quantity=quantity, stop_loss=stop_loss, take_profit=take_profit)
        elif sell_condition:
            quantity = self.calculate_quantity()
            stop_loss = self.calculate_stop_loss(buy=False)
            take_profit = self.calculate_take_profit(buy=False)
            return Signal(action=TradeType.SELL, quantity=quantity, stop_loss=stop_loss, take_profit=take_profit)
        elif current_position != 0 and self.should_close_position(current_position):
            return Signal(action=TradeType.SELL, quantity=abs(current_position), stop_loss=None, take_profit=None)
        else:
            return None

    def should_close_position(self, current_position: float) -> bool:
        """
        The should_close_position function is used to determine whether or not the current position should be closed.

        Args:
            self: Represent the instance of the class
            current_position: float: Determine if the current position is long or short

        Returns:
            True if the current position is greater than 0 and the should_close_long_position function returns true

        """
        if current_position > 0:
            return self.should_close_long_position(current_position)
        elif current_position < 0:
            return self.should_close_short_position(current_position)
        else:
            return False

    def should_close_long_position(self, current_position: float) -> bool:
        """
        The should_close_long_position function is used to determine whether or not a long position should be closed.

        Args:
            self: Represent the instance of the class
            current_position: float: Determine the current position of the stock

        Returns:
            True if the current_position is less than or equal to the stop loss

        """
        return current_position <= self.calculate_stop_loss(buy=True)

    def should_close_short_position(self, current_position: float) -> bool:
        """
        The should_close_short_position function is used to determine whether or not the short position should be closed.
            The function takes in a current_position parameter, which is the current position of the stock.
            The function then returns True if the current_position &gt;= self.calculate_stop_loss(buy=False).
            Otherwise, it returns False.

        Args:
            self: Access the current instance of the class
            current_position: float: Determine the current position of the asset

        Returns:
            True if the current position is greater than or equal to the stop loss

        """
        return current_position >= self.calculate_stop_loss(buy=False)

    def calculate_quantity(self) -> float:
        """
        The calculate_quantity function is used to determine the number of shares to buy or sell.

        Args:
            self: Represent the instance of the class

        Returns:
            The position size
        """
        return self.position_size

    def calculate_stop_loss(self, buy: bool) -> float:
        """
        The calculate_stop_loss function calculates the stop loss price for a given buy or sell order.

        Args:
            self: Represent the instance of the class
            buy: bool: Determine if the user is buying or selling

        Returns:
            A float

        """
        if buy:
            return (1 - self.stop_loss_pct) * self.current_price
        else:
            return (1 + self.stop_loss_pct) * self.current_price

    def calculate_take_profit(self, buy: bool) -> float:
        """
        The calculate_take_profit function calculates the take profit price for a buy or sell order.

        Args:
            self: Represent the instance of the class
            buy: bool: Determine whether the function is calculating a buy or sell take profit

        Returns:
            The price at which the trader should take profit


        """
        if buy:
            return (1 + self.take_profit_pct) * self.current_price
        else:
            return (1 - self.take_profit_pct) * self.current_price


class Position:
    """
    The Position class is used to represent the current position of the stock.
    It is used to keep track of the current
    """

    def __init__(self, symbol, qty, avg_price, side, time):
        self.symbol = symbol
        self.qty = qty
        self.avg_price = avg_price
        self.side = side
        self.time = time


class Rule(BaseModel):
    """
    The Rule class is used to represent a rule in the rule engine.
    """
    name: str
    condition: str
    action: str


class RuleFilter:
    """
    The RuleFilter class is used to filter the rules
     based on the condition and action.
    """
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def apply(self, data):
        """
        The apply function takes a data object and applies the rules to it.
        It does this by iterating through each rule in the rules list, evaluating
        the condition of that rule against the data object, and if true executing
        the action of that rule on the data object.

        Args:
            self: Access the attributes and methods of a class
            data: Pass the data to the rules

        Returns:
            The dataframe with the new column added

        """
        for rule in self.rules:
            if eval(rule.condition):
                exec(rule.action)


from typing import Optional
import alpaca_trade_api as tradeapi


class RuleEngine:
    """
    The RuleEngine class is used to execute the rules.
    """
    def __init__(self, api: tradeapi.REST):
        self.rules = []
        self.api = api

    def place_buy_order(self, symbol: str, qty: float, stop_loss: float, take_profit: float) -> Optional[entity.Order]:
        """
        The place_buy_order function places a buy order for the given symbol, quantity, stop loss and take profit.

        Args:
            self: Represent the instance of the class
            symbol: str: Specify the symbol of the stock you want to buy
            qty: float: Specify the quantity of shares to buy
            stop_loss: float: Set the stop loss for the order
            take_profit: float: Set the take profit price for the order

        Returns:
            An order object

        Doc Author:
            Trelent
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='limit',
                time_in_force='gtc',
                limit_price=self.get_last_trade(symbol).price
            )
            self.calculate_take_profit(order.id, take_profit)
            self.calculate_stop_loss(order.id, stop_loss)
            return order
        except Exception as e:
            print(f"Failed to place buy order for {symbol}: {e}")
            return None

    def calculate_take_profit(self, order_id: str, take_profit: float):
        """
        The calculate_take_profit function takes in an order_id and a take profit percentage.
        It then uses the get_order_by_client_order function to retrieve the order from Binance,
        and if it is filled, calculates a limit price for selling at that take profit level.
        Finally, it submits an order to sell at that limit price.

        Args:
            self: Represent the instance of the class
            order_id: str: Identify the order to be closed
            take_profit: float: Calculate the limit price for the take profit order

        Returns:
            The limit price of the take profit order

        Doc Author:
            Trelent
        """

        try:
            order = self.api.get_order_by_client_order_id(order_id)
            if order and order.status == 'filled':
                limit_price = order.filled_avg_price * (1 + take_profit)
                self.submit_order(
                    order.symbol,
                    order.qty,
                    'sell',
                    'limit',
                    limit_price,
                    f'{order.client_order_id}_tp',
                )
        except Exception as e:
            print(f"Failed to calculate take profit for order {order_id}: {e}")

    def submit_order(self, symbol: str, qty: float, side: str, type: str, price: float, client_order_id: Optional[str] = None) -> Optional[tradeapi.entity.Order]:
        try:
            return self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force='gtc',
                limit_price=price,
                client_order_id=client_order_id,
            )
        except Exception as e:
            print(f"Failed to submit order for {symbol}: {e}")
            return None

    def calculate_stop_loss(self, id, stop_loss):
        """
        The calculate_stop_loss function takes in an order_id and a stop loss percentage.
        It then uses the get_order_by_client_order function to retrieve the order from Binance,
        and if it is filled, calculates a stop price for selling at that stop loss level.
        Finally, it submits an order to sell at that stop price.

        Args:
            self: Represent the instance of the class
            order_id: str: Identify the order to be closed
            stop_loss: float: Calculate the stop price for the stop loss order

        Returns:
            The stop price of the stop loss order

        Doc Author:
            Trelent
        """
        try:
            order = self.api.get_order_by_client_order_id(id)
            if order and order.status == 'filled':
                stop_price = order.filled_avg_price * (1 - stop_loss)
                self.submit_order(
                    order.symbol,
                    order.qty,
                    'sell',
                    'stop',
                    stop_price,
                    f'{order.client_order_id}_sl',
                )
        except Exception as e:
            print(f"Failed to calculate stop loss for order {id}: {e}")

    def add_rule(self, name, condition, actions):
        """
        The add_rule function takes in a name, condition and actions and adds a rule to the rules list.

        Args:
            self: Represent the instance of the class
            name: str: Name of the rule
            condition: str: Condition of the rule
            actions: str: Actions of the rule
        """
        self.rules.append(Rule(name, condition, actions))
        return self.rules

    def get_triggered_rules(self):
        """
        The get_triggered_rules function returns a list of triggered rules.

        Args:
            self: Represent the instance of the class

        Returns:
            A list of triggered rules
        """
        return [rule for rule in self.rules if rule.triggered]

    def get_last_trade(self, symbol):
        """
        The get_last_trade function takes in a symbol and returns the last trade.

        Args:
            self: Represent the instance of the class
            symbol: str: Specify the symbol of the stock you want to get the last trade for

        Returns:
            The last trade for the stock
        """
        return self.api.get_latest_trade(symbol)

    def run(self, conditions):
        """
        The run function takes in a conditions and runs the rules.

        Args:
            self: Represent the instance of the class
            conditions: dict: Specify the conditions for the rules
        """
        for rule in self.rules:
            rule.run(conditions)


def place_buy_order(api, param, qty):
    """
    The place_buy_order function takes in an api, param and quantity and places a buy order.

    Args:
        api: str: Specify the api
        param: str: Specify the param
        qty: str: Specify the quantity
    """
    api.submit_order(
        symbol=param,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='gtc'
    )


def apply_rules(entry_rules: List[Rule], exit_rules: List[Rule], reverse_rules: List[Rule],
                stock_data: List[StockData], positions: List[Position]) -> Tuple[List[Trade], List[Position]]:
    """
    The apply_rules function takes in entry, exit and reverse rules, stock data and positions and applies the rules to the stock data.
    :param entry_rules:
    :param exit_rules:
    :param reverse_rules:
    :param stock_data:
    :param positions:
    """
    # Define conditions for the rules engine
    conditions = {'stock_data': stock_data, 'positions': positions}

    # Create rules engine
    rule_engine = RuleEngine()

    # Add entry rules
    for rule in entry_rules:
        rule_engine.add_rule(rule.name, rule.condition, rule.actions)

    # Add exit rules
    for rule in exit_rules:
        rule_engine.add_rule(rule.name, rule.condition, rule.actions)

    # Add reverse rules
    for rule in reverse_rules:
        rule_engine.add_rule(rule.name, rule.condition, rule.actions)

    # Apply rules
    trades = []
    for i, data in enumerate(stock_data):
        conditions['i'] = i
        rule_engine.run(conditions)

        # Execute actions for entry rules
        if rule_engine.get_triggered_rules():
            for rule in rule_engine.get_triggered_rules():
                if rule.name in [r.name for r in entry_rules]:
                    actions = [a for r in entry_rules if r.name == rule.name for a in r.actions]
                    for action in actions:
                        if action == 'buy':
                            qty = calculate_quantity(api, stock_data[i], positions)
                            order = place_buy_order(api, stock_data[i], qty)
                            position = Position(symbol=stock_data[i].symbol, date=stock_data[i].date, qty=qty,
                                                entry_price=stock_data[i].close, stop_loss=calculate_stop_loss(stock_data[i]),
                                                take_profit=calculate_take_profit(stock_data[i]))
                            positions.append(position)
                            trade = Trade(symbol=stock_data[i].symbol, date=stock_data[i].date, qty=qty,
                                          price=stock_data[i].close, order_type='buy', order=order)
                            trades.append(trade)

        # Execute actions for exit rules
        for position in positions:
            if position.symbol == data.symbol:
                conditions['position'] = position
                rule_engine.run(conditions)

                if rule_engine.get_triggered_rules():
                    for rule in rule_engine.get_triggered_rules():
                        if rule.name in [r.name for r in exit_rules]:
                            actions = [a for r in exit_rules if r.name == rule.name for a in r.actions]
                            for action in actions:
                                if action == 'sell':
                                    order = place_sell_order(api, stock_data[i], position.qty)
                                    trade = Trade(symbol=stock_data[i].symbol, date=stock_data[i].date, qty=position.qty,
                                                  price=stock_data[i].close, order_type='sell', order=order)
                                    trades.append(trade)
                                    positions.remove(position)

    # Reverse positions based on reverse rules
    for position in positions:
        conditions['position'] = position
        rule_engine.run(conditions)

        if rule_engine.get_triggered_rules():
            for rule in rule_engine.get_triggered_rules():
                if rule.name in [r.name for r in reverse_rules]:
                    actions = [a for r in reverse_rules if r.name == rule.name for a in r.actions]
                    for action in actions:
                        if action == 'reverse':
                            close_position(api, position)
                            qty = calculate_quantity(api.get_account().cash, stock_data.close)
                            if qty > 0:
                                if not (
                                        entry_signal := apply_rules(
                                            entry_rules, stock_data, position=None
                                        )
                                ):
                                    continue
                                order = submit_order(api, 'buy', stock_data.symbol, qty, stock_data.close)
                                trade = Trade.from_order(order)
                                current_position = Position.from_trade(trade)
                                exit_signal = apply_rules(exit_rules, stock_data, current_position)
                                reverse_signal = apply_rules(reverse_rules, stock_data, current_position)
                                if exit_signal:
                                    close_position(api, current_position)
                                elif reverse_signal:
                                    close_and_reverse(api, current_position)
                                else:
                                    continue

                            # Define the entry rules


entry_rules = [
    Rule(lambda data: data['bar_num'] % 60 == 59 and data['candle_color'] == 'red', 'buy'),
    Rule(lambda data: data['wicks']['upper'] >= 0.2 and data['wicks']['upper'] < data['ma_25'] * 0.25, 'buy'),
    Rule(lambda data: data['wicks']['upper'] >= 0.3 and data['wicks']['upper'] < data['ma_89'] * 0.25, 'buy'),
    Rule(lambda data: data['candle_color'] == 'red' and data['ma_25'] > data['ma_50'], 'buy'),
    Rule(lambda data: data['ma_25'] > data['ma_50'] and data['ma_50'] > data['ma_89'], 'buy'),
]

# Define the exit rules
exit_rules = [
    Rule(lambda data: data['profit_loss'] >= 0.3 * data['buy_price'], 'sell'),
    Rule(lambda data: data['bar_num'] % 60 == 0 and data['candle_color'] == 'green', 'sell'),
    Rule(lambda data: data['candle_color'] == 'green' and data['wicks']['lower'] >= 0.3 and data['wicks']['lower'] < data['ma_25'] * 0.25, 'sell'),
]

# Define the close/reverse rules
reverse_rules = [
    Rule(lambda data: data['ma_25'] < data['ma_50'] and data['ma_50'] < data['ma_89'], 'reverse'),
    Rule(lambda data: data['ma_25'] < data['ma_50'] and data['profit_loss'] <= -0.2 * data['buy_price'], 'reverse'),
]

# Apply the rules to a stock data dictionary
result = apply_rules(stock_data_dict, entry_rules, exit_rules, reverse_rules)

# The result is a list of trades with the following format:
# [
#   {
#       'symbol': 'AAPL',
#       'timestamp': '2022-05-09T09:30:00-04:00',
#       'action': 'buy',
#       'quantity': 100,
#       'price': 135.0
#   },
#   {
#       'symbol': 'AAPL',
#       'timestamp': '2022-05-09T15:30:00-04:00',
#       'action': 'sell',
#       'quantity': 100,
#       'price': 137.5
#   }
# ]
