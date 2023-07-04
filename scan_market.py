from typing import List
from actors import StockDataActor, RuleActor
from models import StockData, Rule


def scan_market(stocks: List[str], stock_data_actors: List[StockDataActor], rule_actor: RuleActor) -> List[str]:
    """
    Scans the market and selects symbols to trade based on the filter criteria.

    Parameters:
    stocks (List[str]): List of stock symbols to scan.
    stock_data_actors (List[StockDataActor]): List of actors for retrieving stock data.
    rule_actor (RuleActor): Actor for applying trading rules.

    Returns:
    List[str]: List of symbols to trade.
    """

    # Create a dictionary to store the latest stock data for each symbol
    latest_data = {}

    # Retrieve the latest stock data for each symbol using the StockDataActor instances
    for stock in stocks:
        actor = next(actor for actor in stock_data_actors if actor.symbol == stock)
        latest_data[stock] = actor.get_latest_data()

    # Retrieve the current trade from the RuleActor
    current_trade = rule_actor.get_current_trade()

    # Retrieve the trading rules from the RuleActor
    rules = rule_actor.get_rules()

    # Apply the trading rules to each stock and select the ones that match
    trade_candidates = []
    for stock, data in latest_data.items():
        # Check if the current trade is already open for this stock
        if current_trade and current_trade.stock == stock:
            continue

        # Apply the rules to the stock data using the RuleActor
        is_candidate = rule_actor.apply_rules(data, rules)

        if is_candidate:
            trade_candidates.append(stock)

    return trade_candidates
