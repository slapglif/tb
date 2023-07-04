from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_mixins import AllFeaturesMixin

Base = declarative_base()
from typing import List
import numpy as np

from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class TradingRule(Base):
    __tablename__ = 'trading_rules'

    id = Column(Integer, primary_key=True)
    rule_name = Column(String)
    entry_rule = Column(String)
    exit_rule = Column(String)
    close_reverse_rule = Column(String)

    def __repr__(self):
        return f"<TradingRule(id={self.id}, rule_name='{self.rule_name}')>"

class TradingRulesDB:
    def __init__(self, db_url: str):
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def add_rule(self, rule_name: str, entry_rule: str, exit_rule: str, close_reverse_rule: str) -> None:
        trading_rule = TradingRule(rule_name=rule_name, entry_rule=entry_rule, exit_rule=exit_rule, close_reverse_rule=close_reverse_rule)
        self.session.add(trading_rule)
        self.session.commit()

    def get_rule(self, rule_name: str) -> TradingRule:
        return self.session.query(TradingRule).filter_by(rule_name=rule_name).first()

    def get_all_rules(self) -> List[TradingRule]:
        return self.session.query(TradingRule).all()

    def update_rule(self, rule_name: str, entry_rule: str, exit_rule: str, close_reverse_rule: str) -> None:
        trading_rule = self.get_rule(rule_name)
        trading_rule.entry_rule = entry_rule
        trading_rule.exit_rule = exit_rule
        trading_rule.close_reverse_rule = close_reverse_rule
        self.session.commit()



class TradeHistory(Base):
    __tablename__ = 'trade_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    time = Column(DateTime, nullable=False)
    qty = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    side = Column(String, nullable=False)
    order_id = Column(String, nullable=False)

    def __repr__(self):
        return f"TradeHistory(id={self.id}, symbol='{self.symbol}', time='{self.time}', qty={self.qty}, price={self.price}, side='{self.side}', order_id='{self.order_id}')"


def calculate_sharpe_ratio(profits: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio of a trading strategy given a list of profits and a risk-free rate.
    """
    returns = np.array(profits)
    excess_returns = returns - risk_free_rate
    mean_excess_returns = excess_returns.mean()
    std_excess_returns = excess_returns.std()

    if std_excess_returns == 0:
        return 0

    return mean_excess_returns / std_excess_returns


def calculate_sortino_ratio(profits: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sortino ratio of a trading strategy given a list of profits and a risk-free rate.
    """
    returns = np.array(profits)
    excess_returns = returns - risk_free_rate
    downside_returns = np.where(excess_returns < 0, excess_returns, 0)
    mean_downside_returns = downside_returns.mean()
    std_downside_returns = downside_returns.std()

    if std_downside_returns == 0:
        return 0

    sortino_ratio = (returns.mean() - risk_free_rate) / std_downside_returns

    return sortino_ratio


def calculate_max_drawdown(profits: List[float]) -> float:
    """
    Calculate the maximum drawdown of a trading strategy given a list of profits.
    """
    returns = np.array(profits)
    cumulative_returns = (1 + returns).cumprod()
    max_return = cumulative_returns.cummax()
    drawdown = (cumulative_returns - max_return) / max_return
    max_drawdown = abs(drawdown.min())

    return max_drawdown


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class StockData(Base):
    __tablename__ = 'stock_data'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    date = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    vwap = Column(Float)
    adj_close = Column(Float)


class TradingStats(Base):
    __tablename__ = 'trading_stats'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    date = Column(DateTime)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    avg_profit = Column(Float)
    max_profit = Column(Float)
    min_profit = Column(Float)
    avg_loss = Column(Float)
    max_loss = Column(Float)
    min_loss = Column(Float)
    net_profit = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)

    @classmethod
    def fetch_data_from_api(symbol: str, start_date: datetime, end_date: datetime) -> List[dict]:
        """
        Fetches historical stock data from the Alpaca API for a specific stock within a date range.
        """
        timeframe = '1D'
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        stock_data = api.get_barset(symbol, timeframe, start=start_date, end=end_date).df[symbol]
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={'time': 'date'}, inplace=True)
        stock_data['date'] = stock_data['date'].dt.date
        stock_data['vwap'] = (stock_data['volume'] * (stock_data['open'] + stock_data['high'] + stock_data['low'] + stock_data['close'])) / (4 * stock_data['volume'])
        stock_data.drop(columns=['time', 'currency'], inplace=True)
        stock_data = stock_data.to_dict('records')
        return stock_data

def store_data_to_db(stock_data: pd.DataFrame, trading_stats: dict, db_conn: str):
    engine = create_engine(db_conn)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for i, row in stock_data.iterrows():
        stock = StockData(symbol=row['symbol'],
                          date=row['date'],
                          open=row['open'],
                          high=row['high'],
                          low=row['low'],
                          close=row['close'],
                          volume=row['volume'],
                          vwap=row['vwap'],
                          adj_close=row['adj_close'])
        session.add(stock)

    for symbol, stats in trading_stats.items():
        trading = TradingStats(symbol=symbol,
                               date=stats['date'],
                               total_trades=stats['total_trades'],
                               win_rate=stats['win_rate'],
                               avg_profit=stats['avg_profit'],
                               max_profit=stats['max_profit'],
                               min_profit=stats['min_profit'],
                               avg_loss=stats['avg_loss'],
                               max_loss=stats['max_loss'],
                               min_loss=stats['min_loss'],
                               net_profit=stats['net_profit'],
                               sharpe_ratio=stats['sharpe_ratio'],
                               sortino_ratio=stats['sortino_ratio'],
                               max_drawdown=stats['max_drawdown'])
        session.add(trading)

    session.commit()


class SymbolStat(Base, TimestampMixin):
    __tablename__ = 'symbol_stat'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    open_price = Column(Float)
    close_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    volume = Column(Float)
    vwap = Column(Float)
    ema20 = Column(Float)
    ema50 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    signal = Column(Float)
    histogram = Column(Float)
    prev_day_close_price = Column(Float)
    prev_day_volume = Column(Float)
    prev_day_vwap = Column(Float)


    @classmethod
    def from_barset(cls, symbol: str, barset: BarSet):
        bars = barset[symbol]
        stats = {
            'symbol': symbol,
            'open_price': bars[0].o,
            'close_price': bars[-1].c,
            'high_price': max(bar.h for bar in bars),
            'low_price': min(bar.l for bar in bars),
            'volume': sum(bar.v for bar in bars),
            'vwap': np.average([bar.c for bar in bars], weights=[bar.v for bar in bars]),
            'prev_day_close_price': bars[-2].c,
            'prev_day_volume': bars[-2].v,
            'prev_day_vwap': bars[-2].vw,
        }
        ema20 = ta.ema(np.array([bar.c for bar in bars]), timeperiod=20)
        ema50 = ta.ema(np.array([bar.c for bar in bars]), timeperiod=50)
        stats['ema20'] = ema20[-1]
        stats['ema50'] = ema50[-1]
        rsi = ta.rsi(np.array([bar.c for bar in bars]))
        stats['rsi'] = rsi[-1]
        macd, signal, histogram = ta.macd(np.array([bar.c for bar in bars]))
        stats['macd'] = macd[-1]
        stats['signal'] = signal[-1]
        stats['histogram'] = histogram[-1]
        return cls(**stats)

    @classmethod
    def get_latest_stat(cls, symbol: str, session: Session) -> Optional['SymbolStat']:
        return session.query(cls).filter_by(symbol=symbol).order_by(cls.timestamp.desc()).first()

    @classmethod
    def validate_stat(cls, symbol: str, stat: Dict[str, Any]) -> bool:
        if stat['symbol'] != symbol:
            return False
        if not isinstance(stat['open_price'], (int, float)):
            return False
        if not isinstance(stat['close_price'], (int, float)):
            return False
        if not isinstance(stat['high_price'], (int, float)):
            return False
        if not isinstance(stat['low_price'], (int, float)):
            return False
        if not isinstance(stat['volume'], (int, float)):
            return False
        if not isinstance(stat['vwap'], (int, float)):
            return False
        if not isinstance(stat['ema20'], (int, float)):
            return False
        if not isinstance(stat['ema50'], (int, float)):
            return False
        if not isinstance(stat['rsi'], (int, float)):
            return False
        if not isinstance(stat['macd'], (int, float)):
            return False
        if not isinstance(stat['signal'], (int, float)):
            return False
        if not isinstance(stat['histogram'], (int, float)):
            return False
        if not isinstance(stat['prev_day_close_price'], (int, float)):
            return False
        return (
            isinstance(stat['prev_day_vwap'], (int, float))
            if isinstance(stat['prev_day_volume'], (int, float))
            else False
        )


