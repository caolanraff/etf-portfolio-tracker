"""Utility functions."""
from datetime import datetime

from src.utils.types import Time


def parse_date(date_str: str, default_date: Time) -> Time:
    """Parse date string or return default date."""
    return datetime.strptime(date_str, "%Y-%m-%d") if date_str else default_date
