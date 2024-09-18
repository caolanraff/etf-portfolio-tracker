"""Utility functions."""
from datetime import datetime

from src.utils.types import Time


def parse_date(date_str: str, default_date: Time) -> Time:
    """Parse date string or return default date."""
    return datetime.strptime(date_str, "%Y-%m-%d") if date_str else default_date


def convert_to_snake_case(text: str) -> str:
    """Convert text to snake case."""
    return text.lower().replace(" ", "_")
