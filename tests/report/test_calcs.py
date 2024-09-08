import pandas as pd
from pandas.testing import assert_frame_equal

from src.report.calcs import calculate_entry_price


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
