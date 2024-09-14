import pandas as pd
from pandas.testing import assert_frame_equal

from src.report.calcs import calculate_entry_price, process_stock_splits


def test_calculate_entry_price() -> None:
    data = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "quantity": [10, 20, 30],
            "price": [100, 150, 200],
        }
    )

    result = calculate_entry_price(data).round(2)
    expected = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "average_entry_price": [100.0, 133.33, 166.67],
        }
    )

    assert_frame_equal(result, expected)


def test_correctly_applies_stock_splits() -> None:
    data = pd.DataFrame(
        {
            "ticker": ["SOXX", "SOXX", "AAPL"],
            "date": ["2024-03-05", "2024-03-07", "2024-03-08"],
            "quantity": [100, 100, 50],
            "price": [300, 300, 150],
        }
    )
    stock_splits = pd.DataFrame(
        {"ticker": ["SOXX"], "date": ["2024-03-07"], "ratio": [3]}
    )

    result = process_stock_splits(data, stock_splits)
    expected = pd.DataFrame(
        {
            "ticker": ["SOXX", "SOXX", "AAPL"],
            "date": ["2024-03-05", "2024-03-07", "2024-03-08"],
            "quantity": [300, 300, 50],
            "price": [100, 100, 150],
        }
    )

    assert_frame_equal(result, expected)
