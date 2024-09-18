from datetime import datetime

from src.utils.util import convert_to_snake_case, parse_date


def test_parse_date() -> None:
    date_str = "2023-10-01"
    default_date = datetime(2023, 1, 1)

    result = parse_date(date_str, default_date)
    expected = datetime(2023, 10, 1)

    assert result == expected


def test_convert_to_snake_case() -> None:
    result = convert_to_snake_case("New Trades")
    expected = "new_trades"

    assert result == expected
