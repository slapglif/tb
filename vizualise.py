import seaborn as sns
import matplotlib.pyplot as plt

from rules import TradeType


def visualize_results(stock_data, trade_history, performance_stats):
    # Plot stock data with VWAP and trade signals
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['Close'], label='Price')
    ax.plot(stock_data['vwap'], label='VWAP')
    ax.plot(trade_history.loc[trade_history['signal'] == 'BUY'].index,
            stock_data.loc[trade_history['signal'] == 'BUY', 'Close'],
            '^', markersize=10, color='green', label='Buy signal')
    ax.plot(trade_history.loc[trade_history['signal'] == 'SELL'].index,
            stock_data.loc[trade_history['signal'] == 'SELL', 'Close'],
            'v', markersize=10, color='red', label='Sell signal')
    ax.legend()

    # Plot performance stats with bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=performance_stats.index, y=performance_stats.values, ax=ax)
    ax.set_ylabel('Value')
    ax.set_title('Performance Stats')

    # Plot equity curve
    equity_curve = trade_history['pnl'].cumsum()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_curve, label='Equity curve')
    ax.plot(trade_history.loc[trade_history['trade_type'] == TradeType.BUY].index,
            equity_curve.loc[trade_history['trade_type'] == TradeType.BUY],
            '^', markersize=10, color='green', label='Buy trade')
    ax.plot(trade_history.loc[trade_history['trade_type'] == TradeType.SELL].index,
            equity_curve.loc[trade_history['trade_type'] == TradeType.SELL],
            'v', markersize=10, color='red', label='Sell trade')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('Equity Curve')

    # Show plots
    plt.show()
