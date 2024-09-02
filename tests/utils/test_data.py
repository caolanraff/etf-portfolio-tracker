from datetime import date
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.utils.data import get_ticker_data, get_ticker_info, ticker_data, ticker_info


def test_get_ticker_data(mocker: Any) -> None:
    mock_data = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [110, 111, 112],
            "Low": [90, 91, 92],
            "Close": [105, 106, 107],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range(start="2023-01-01", periods=3),
    )

    mocker.patch("yfinance.download", return_value=mock_data)

    ticker = "AAPL"
    result = get_ticker_data(ticker)

    expected = mock_data.reindex(
        pd.date_range(min(list(mock_data.index)), date.today(), freq="D")
    )
    expected = expected.ffill()

    assert not result.empty
    assert ticker in ticker_data
    assert_frame_equal(result, expected)
    assert result.loc["2023-01-01"]["Close"] == 105
    assert result.loc[np.datetime64(date.today())]["Close"] == 107


class TestObject:
    @property
    def info(self) -> Dict[str, str]:
        return {"symbol": "AAPL", "name": "Apple Inc."}


def test_get_ticker_info(mocker: Any) -> None:
    ticker = "AAPL"
    mocker.patch("yfinance.Ticker", return_value=TestObject())

    result = get_ticker_info(ticker)

    expected = {"symbol": "AAPL", "name": "Apple Inc."}

    assert result
    assert ticker in ticker_info
    assert result == expected
