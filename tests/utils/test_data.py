from datetime import date
from typing import Any, Dict

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.utils.data import (
    get_anchor_from_html,
    get_etf_underlyings,
    get_ticker_data,
    get_ticker_info,
    get_title_from_html,
    ticker_data,
    ticker_info,
)


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

    result = get_ticker_data("VONG")
    expected = mock_data.reindex(
        pd.date_range(min(list(mock_data.index)), date.today(), freq="D")
    )
    expected = expected.ffill()

    assert "VONG" in ticker_data
    assert_frame_equal(result, expected)

    # test cache
    result = get_ticker_data("VONG")
    assert_frame_equal(result, expected)

    # test yfinance failure
    mocker.patch("yfinance.download", side_effect=Exception("Custom Error Message"))
    with pytest.raises(SystemExit):
        get_ticker_data("ABC")

    # test no data
    mocker.patch("yfinance.download", return_value=pd.DataFrame())
    with pytest.raises(SystemExit):
        get_ticker_data("DEF")


class TestObject:
    @property
    def info(self) -> Dict[str, str]:
        return {"symbol": "AAPL", "name": "Apple Inc."}


def test_get_ticker_info(mocker: Any) -> None:
    mocker.patch("yfinance.Ticker", return_value=TestObject())

    result = get_ticker_info("AAPL")
    expected = {"symbol": "AAPL", "name": "Apple Inc."}

    assert "AAPL" in ticker_info
    assert result == expected

    # test cache
    result = get_ticker_info("AAPL")
    assert result == expected

    # test yfinance failure
    mocker.patch("yfinance.Ticker", side_effect=Exception("Custom Error Message"))
    with pytest.raises(SystemExit):
        get_ticker_info("ABC")


def test_get_title_from_html() -> None:
    html_string = '<div title="Example Title - Subtitle"></div>'

    result = get_title_from_html(html_string)
    expected = "Example Title"

    assert result == expected


def test_get_anchor_from_html() -> None:
    html_string = '<a href="https://example.com" rel="VONG">Example</a>'
    result = get_anchor_from_html(html_string)
    assert result == "VONG"

    # short string
    result = get_anchor_from_html("ABC")
    assert result == "ABC"

    # no a tag
    html_string = '<z href="https://example.com" rel="VONG">Example</z>'
    result = get_anchor_from_html(html_string)
    assert result == ""


def test_get_etf_underlyings(mocker: Any) -> None:
    mock_response = mocker.Mock()
    mock_response.text = """
        etf_holdings.formatted_data = [
            ["Apple Inc.", "<a rel='AAPL'>AAPL</a>", "Technology", "10.0"],
            ["Microsoft Corp.", "<a rel='MSFT'>MSFT</a>", "Technology", "20.0"]
        ];
    """
    mocker.patch("requests.Session.get", return_value=mock_response)
    mocker.patch(
        "src.utils.data.get_ticker_info", return_value={"category": "Technology"}
    )

    result = get_etf_underlyings("SPY")
    expected = pd.DataFrame(
        {
            "ticker": ["SPY", "SPY"],
            "Stock": ["AAPL", "MSFT"],
            "Company": ["Apple Inc.", "Microsoft Corp."],
            "Weight": [10.0, 20.0],
        }
    )
    assert_frame_equal(result, expected)

    # test cache
    result = get_etf_underlyings("SPY")
    assert_frame_equal(result, expected)

    # test exception
    mocker.patch("json.loads", side_effect=Exception("JSON decoding error"))
    result = get_etf_underlyings("VOO")
    assert_frame_equal(result, pd.DataFrame())


def test_get_etf_underlyings_bond(mocker: Any) -> None:
    mock_response = mocker.Mock()
    mock_response.text = """
        etf_holdings.formatted_data = [
            ["Apple Inc.", "<a rel='AAPL'>AAPL</a>", "Bond", "10.0"],
            ["Microsoft Corp.", "<a rel='MSFT'>MSFT</a>", "Bond", "20.0"]
        ];
    """
    mocker.patch("requests.Session.get", return_value=mock_response)
    mocker.patch("src.utils.data.get_ticker_info", return_value={"category": " bond"})

    result = get_etf_underlyings("BIV")
    expected = pd.DataFrame(
        {
            "ticker": ["BIV", "BIV"],
            "Stock": ["AAPL", "MSFT"],
            "Company": ["Apple Inc. (Bond)", "Microsoft Corp. (Bond)"],
            "Weight": [10.0, 20.0],
        }
    )
    assert_frame_equal(result, expected)
