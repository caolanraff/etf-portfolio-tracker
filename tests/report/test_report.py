from datetime import datetime

import pandas as pd

from src.report.report import get_aum
from src.utils.types import DictFrame


def test_get_aum() -> None:
    result_dict: DictFrame = {
        "Portfolio1": pd.DataFrame(
            {
                "date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "cumulative_quantity": [10, 20],
                "portfolio_value": [1000, 2000],
            }
        ),
        "Portfolio2": pd.DataFrame(
            {
                "date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "cumulative_quantity": [5, 15],
                "portfolio_value": [500, 1500],
            }
        ),
    }
    end_date = datetime(2023, 1, 2)

    result = get_aum(result_dict, end_date)
    expected = "$3,500"

    assert result == expected
