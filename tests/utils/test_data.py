from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.utils.data import get_ticker_data, get_ticker_info, ticker_data


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

    assert not result.empty
    assert ticker in ticker_data
    assert result.equals(ticker_data[ticker])

    expected_index = pd.date_range(start="2023-01-01", end=date.today(), freq="D")
    assert result.index.equals(expected_index)
    assert result.loc["2023-01-01"]["Close"] == 105
    assert result.loc[np.datetime64(date.today())]["Close"] == 107


def test_get_ticker_info(mocker: Any) -> None:
    ticker = "AAPL"
    expected_data = {"symbol": "AAPL", "name": "Apple Inc."}
    mocker.patch("yfinance.Ticker.info", return_value=expected_data)

    result = get_ticker_info(ticker)
    result = result.return_value

    assert result == expected_data
