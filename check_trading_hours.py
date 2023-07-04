from typing import Tuple
import datetime


def check_trading_hours(current_time: datetime.datetime, trading_hours: Tuple[str, str]) -> bool:
    """
    Checks if the current time is within the specified trading hours.

    :param current_time: A datetime object representing the current time.
    :param trading_hours: A tuple containing the start and end times of the trading hours (in the format 'HH:MM').
    :return: A boolean indicating whether the current time is within the specified trading hours.
    """
    start_time_str, end_time_str = trading_hours
    start_time = datetime.time.fromisoformat(start_time_str)
    end_time = datetime.time.fromisoformat(end_time_str)

    return start_time <= current_time.time() <= end_time
