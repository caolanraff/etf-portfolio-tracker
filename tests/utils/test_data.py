from datetime import date
from typing import Any, Dict

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.report.errors import NoDataErr
from src.utils.data import (
    get_anchor_from_html,
    get_etf_underlyings,
    get_metrics,
    get_sector_weightings,
    get_ticker_data,
    get_ticker_info,
    get_ticker_metrics,
    get_title_from_html,
    ticker_data,
)


def test_get_ticker_data(mocker: Any) -> None:
    price_metrics = ["Open", "High", "Low", "Close", "Volume"]
    multi_index = pd.MultiIndex.from_product(
        [price_metrics, ["VONG"]], names=["Price", "Ticker"]
    )
    mock_data = pd.DataFrame(
        [
            [100, 110, 90, 105, 1000],
            [101, 111, 91, 106, 1100],
            [102, 112, 92, 107, 1200],
        ],
        index=pd.date_range(start="2023-01-01", periods=3),
        columns=multi_index,
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


class TickerInfoTestObject:
    @property
    def info(self) -> Dict[str, str]:
        return {"symbol": "AAPL", "name": "Apple Inc."}


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
    # test external
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

    result = get_etf_underlyings(["SPY"], "external")
    expected = pd.DataFrame(
        {
            "ticker": ["SPY", "SPY"],
            "Stock": ["AAPL", "MSFT"],
            "Company": ["Apple Inc.", "Microsoft Corp."],
            "Weight": [10.0, 20.0],
        }
    )
    assert_frame_equal(result, expected)

    # test internal
    mocker.patch("pandas.read_csv", return_value=expected)
    result = get_etf_underlyings(["SPY"], "internal")
    assert_frame_equal(result, expected)

    # test cache
    result = get_etf_underlyings(["SPY"], "external")
    assert_frame_equal(result, expected)

    # test exception
    with pytest.raises(ValueError):
        get_etf_underlyings(["VOO"], "other")

    mocker.patch("json.loads", side_effect=Exception("JSON decoding error"))
    with pytest.raises(NoDataErr):
        get_etf_underlyings(["VOO"], "external")

    mocker.patch("json.loads", return_value=[["NA", "NA", "NA", "NA", "NA"]])
    with pytest.raises(NoDataErr):
        get_etf_underlyings(["VOO"], "external")


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

    result = get_etf_underlyings(["BIV"], "external")
    expected = pd.DataFrame(
        {
            "ticker": ["BIV", "BIV"],
            "Stock": ["AAPL", "MSFT"],
            "Company": ["Apple Inc. (Bond)", "Microsoft Corp. (Bond)"],
            "Weight": [10.0, 20.0],
        }
    )
    assert_frame_equal(result, expected)


class TickerTestObject:
    @property
    def all_modules(self) -> Any:
        return {"VOO": {"assetProfile": {"longBusinessSummary": "fund"}}}


class TickerTestObjectErr:
    @property
    def all_modules(self) -> Any:
        return {"ABC": "Invalid Crumb"}


def test_get_ticker_metrics(mocker: Any) -> None:
    mocker.patch("yahooquery.Ticker", return_value=TickerTestObject())

    result = get_ticker_metrics("VOO")
    expected = TickerTestObject().all_modules["VOO"]
    assert result == expected

    # test cache
    result = get_ticker_metrics("VOO")
    assert result == expected

    # test failure
    mocker.patch("yahooquery.Ticker", return_value=TickerTestObjectErr())
    with pytest.raises(RuntimeError):
        get_ticker_metrics("ABC", 2, 0)

    mocker.patch("yahooquery.Ticker", side_effect=Exception("Custom Error Message"))
    with pytest.raises(NoDataErr):
        get_ticker_metrics("ABC")


def test_get_metrics(mocker: Any) -> None:
    metrics = {
        "fundProfile": {"feesExpensesInvestment": {"annualReportExpenseRatio": 0.0001}},
        "summaryDetail": {"trailingPE": 25.0, "yield": 0.007, "volume": 100000},
        "defaultKeyStatistics": {
            "ytdReturn": 0.051,
            "beta3Year": 1.1,
            "totalAssets": 1000000000.0,
            "threeYearAverageReturn": 0.17,
        },
        "fundPerformance": {
            "riskOverviewStatistics": {"riskStatistics": [{"sharpeRatio": 1.3}]}
        },
    }
    mocker.patch("src.utils.data.get_ticker_metrics", return_value=metrics)

    result = get_metrics(["VOO"])
    expected = pd.DataFrame(
        [
            {
                "Ticker": "VOO",
                "Exp. Ratio": 0.01,
                "Div. Yield": 0.7,
                "Sharpe Ratio": 1.3,
                "Beta": 1.1,
                "PE Ratio": 25.0,
                "Volume": 100000,
                "Assets": 1.0,
                "YTD Return": 5.1,
                "3yr Return": 17.0,
            }
        ]
    )

    assert_frame_equal(result, expected)


def test_get_sector_weightings(mocker: Any) -> None:
    weights = [
        {"basic_materials": 0.0173},
        {"consumer_defensive": 0.0547},
        {"technology": 0.3199},
        {"financial_services": 0.1361},
        {"utilities": 0.0253},
        {"energy": 0.0315},
        {"healthcare": 0.104899995},
    ]
    metrics = {"topHoldings": {"sectorWeightings": weights}}
    mocker.patch("src.utils.data.get_ticker_metrics", return_value=metrics)

    result = get_sector_weightings(["VOO"])
    expected = pd.DataFrame(
        [(k, v) for d in weights for k, v in d.items()], columns=["Sector", "Weight"]
    )
    expected.insert(0, "Ticker", "VOO")

    assert_frame_equal(result, expected)


def test_get_ticker_info(mocker: Any) -> None:
    metrics = {
        "price": {"shortName": "S&P 500"},
        "fundProfile": {"categoryName": "Index Fund"},
        "summaryProfile": {"longBusinessSummary": "S&P 500 index fund."},
    }
    mocker.patch("src.utils.data.get_ticker_metrics", return_value=metrics)

    result = get_ticker_info("VOO")
    expected = {
        "name": "S&P 500",
        "category": "Index Fund",
        "description": "S&P 500 index fund.",
    }

    assert result == expected
