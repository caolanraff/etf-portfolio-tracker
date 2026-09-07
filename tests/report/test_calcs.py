from datetime import datetime
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from src.report.calcs import (
    calculate_all_portfolio_pnl,
    calculate_costs_and_proceeds,
    calculate_entry_price,
    calculate_portfolio_pnl,
    calculate_portfolio_pnl_history,
    calculate_portfolio_risk_metrics,
    process_stock_splits,
)


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


def test_calculate_costs_and_proceeds() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "ticker": ["AAPL"] * 5,
            "quantity": [10, -5, 15, -10, 5],
            "price": [150, 155, 160, 165, 170],
        }
    )

    result = calculate_costs_and_proceeds("AAPL", data, "2023-01-05")
    result = result["cumulative_quantity"].tolist()
    expected = [10, 5, 20, 10, 15]

    assert result == expected


def test_process_stock_splits() -> None:
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

    # no stock splits
    result = process_stock_splits(data, pd.DataFrame(columns=["ticker"]))
    expected = data
    assert_frame_equal(result, expected)


def test_calculate_portfolio_pnl(mocker: Any) -> None:
    ticker_data = pd.DataFrame(
        {"Close": [100.0, 105.0, 110.0]},
        index=pd.date_range(start="2023-01-01", periods=3),
    )
    mocker.patch("src.report.calcs.get_ticker_data", return_value=ticker_data)

    data = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-03"),
            ],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "quantity": [10, -5, 5],
            "price": [100.0, 105.0, 110.0],
        }
    )

    result = calculate_portfolio_pnl(data, datetime(2023, 1, 3))
    expected = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-03"),
            ],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "quantity": [10.0, -5.0, 0.0],
            "price": [100.0, 105.0, 0.0],
            "cumulative_quantity": [10.0, 5.0, 5.0],
            "total_cost": [1000.0, 0.0, 0.0],
            "cumulative_cost": [1000.0, 1000.0, 1000.0],
            "total_proceeds": [0.0, -525.0, 0.0],
            "cumulative_proceeds": [0.0, -525.0, -525.0],
            "average_entry_price": [100.0, 100.0, 100.0],
            "market_price": [100.0, 105.0, 110.0],
            "notional_value": [1000.0, 525.0, 550.0],
            "unrealised_pnl": [0.0, 25.0, 50.0],
            "realised_pnl": [0.0, 25.0, 25.0],
            "total_pnl": [0.0, 50.0, 75.0],
            "portfolio_pnl": [0.0, 50.0, 75.0],
            "portfolio_cost": [1000.0, 1525.0, 1525.0],
            "portfolio_value": [1000.0, 525.0, 550.0],
            "pnl_pct": [0.0, 3.278689, 4.918033],
        }
    )

    assert_frame_equal(result, expected)


def test_calculate_portfolio_risk_metrics() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "portfolio_pnl": [0.0, 20.0, 50.0, 10.0, 30.0],
            "portfolio_value": [1000.0] * 5,
            "total_cost": [0.0] * 5,
        }
    )

    result = calculate_portfolio_risk_metrics(data, datetime(2023, 1, 5))
    expected = {"Volatility": 50.82, "Sharpe Ratio": 3.72, "Max Drawdown": -3.81}

    assert result == expected


def test_calculate_portfolio_risk_metrics_flat() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=3, freq="D"),
            "portfolio_pnl": [0.0, 0.0, 0.0],
            "portfolio_value": [1000.0] * 3,
            "total_cost": [0.0] * 3,
        }
    )

    result = calculate_portfolio_risk_metrics(data, datetime(2023, 1, 3))
    expected = {"Volatility": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0}

    assert result == expected


def test_calculate_portfolio_risk_metrics_trailing_window() -> None:
    # A large loss over 2 years ago, then 5 flat trailing days: only the trailing
    # window should be used, so the old loss shouldn't show up in the drawdown/vol.
    old_dates = pd.date_range(start="2020-01-01", periods=3, freq="D")
    recent_dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "date": list(old_dates) + list(recent_dates),
            "portfolio_pnl": [0.0, -500.0, 0.0] + [0.0] * 5,
            "portfolio_value": [1000.0] * 8,
            "total_cost": [0.0] * 8,
        }
    )

    result = calculate_portfolio_risk_metrics(data, datetime(2023, 1, 5))
    expected = {"Volatility": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0}

    assert result == expected


def test_calculate_portfolio_risk_metrics_fallback_to_inception() -> None:
    # A portfolio younger than the lookback window should just use its full history.
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "portfolio_pnl": [0.0, 20.0, 50.0, 10.0, 30.0],
            "portfolio_value": [1000.0] * 5,
            "total_cost": [0.0] * 5,
        }
    )

    result = calculate_portfolio_risk_metrics(
        data, datetime(2023, 1, 5), lookback_days=10000
    )
    expected = {"Volatility": 50.82, "Sharpe Ratio": 3.72, "Max Drawdown": -3.81}

    assert result == expected


def test_calculate_portfolio_pnl_history(mocker: Any) -> None:
    mock_excel = mocker.patch("pandas.ExcelFile")
    mock_read_excel = mocker.patch("pandas.read_excel")

    mock_excel.return_value.sheet_names = ["Portfolio1", "Portfolio2"]
    mock_data1 = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "ticker": ["AAPL", "AAPL"],
            "quantity": [10, 5],
            "price": [150, 155],
        }
    )
    mock_read_excel.side_effect = [mock_data1, pd.DataFrame()]

    mock_calculate_pnl = mocker.patch("src.report.calcs.calculate_portfolio_pnl")
    mock_calculate_pnl.side_effect = [mock_data1]

    result = calculate_portfolio_pnl_history("dummy_path.xlsx", datetime(2023, 1, 31))

    assert list(result.keys()) == ["Portfolio1"]
    assert_frame_equal(result["Portfolio1"], mock_data1)


def test_calculate_all_portfolio_pnl(mocker: Any) -> None:
    mock_excel = mocker.patch("pandas.ExcelFile")
    mock_read_excel = mocker.patch("pandas.read_excel")

    mock_excel.return_value.sheet_names = ["Portfolio1", "Portfolio2", "Portfolio3"]
    mock_data1 = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "ticker": ["AAPL", "AAPL"],
            "quantity": [10, 5],
            "price": [150, 155],
        }
    )
    mock_data2 = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "ticker": ["GOOGL", "GOOGL"],
            "quantity": [8, 4],
            "price": [1000, 1020],
        }
    )
    mock_read_excel.side_effect = [mock_data1, mock_data2, pd.DataFrame()]

    mock_calculate_pnl = mocker.patch("src.report.calcs.calculate_portfolio_pnl")
    mock_data1["cumulative_quantity"] = mock_data1["quantity"].cumsum()
    mock_data2["cumulative_quantity"] = mock_data2["quantity"].cumsum()
    mock_calculate_pnl.side_effect = [mock_data1, mock_data2]

    result = calculate_all_portfolio_pnl(
        "dummy_path.xlsx", "2023-01-01", "2023-01-31", ""
    )

    assert "Portfolio1" in result
    assert "Portfolio2" in result
    assert_frame_equal(result["Portfolio1"], mock_data1)
    assert_frame_equal(result["Portfolio2"], mock_data2)


def test_calculate_all_portfolio_pnl_drops_near_zero_position(mocker: Any) -> None:
    # A closed-out position can sum to a tiny floating-point residue (e.g. 7.1e-15)
    # instead of exactly 0, from summing many non-exact decimal trade quantities.
    # It should still be dropped, not kept around as a phantom position.
    mock_excel = mocker.patch("pandas.ExcelFile")
    mock_read_excel = mocker.patch("pandas.read_excel")

    mock_excel.return_value.sheet_names = ["Portfolio1"]
    mock_data = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "ticker": ["AAPL", "GHOST", "AAPL", "GHOST"],
            "quantity": [10, 0.1, 5, -0.1],
        }
    )
    mock_read_excel.side_effect = [mock_data]

    pnl_data = mock_data.copy()
    pnl_data["cumulative_quantity"] = [10.0, 7.1e-15, 15.0, -7.1e-15]
    mocker.patch("src.report.calcs.calculate_portfolio_pnl", return_value=pnl_data)

    result = calculate_all_portfolio_pnl(
        "dummy_path.xlsx", "2023-01-01", "2023-01-31", ""
    )

    assert result["Portfolio1"]["ticker"].unique().tolist() == ["AAPL"]


def test_calculate_all_portfolio_pnl_benchmark(mocker: Any) -> None:
    mock_excel = mocker.patch("pandas.ExcelFile")
    mock_excel.return_value.sheet_names = []

    ticker_data = pd.DataFrame(
        {"Close": [100.0, 105.0, 110.0]},
        index=pd.date_range(start="2023-01-29", periods=3),
    )
    mocker.patch("src.report.calcs.get_ticker_data", return_value=ticker_data)

    mock_calculate_pnl = mocker.patch("src.report.calcs.calculate_portfolio_pnl")
    mock_calculate_pnl.side_effect = [ticker_data]

    result = calculate_all_portfolio_pnl(
        "dummy_path.xlsx", "2023-01-01", "2023-01-31", "SPY"
    )

    assert "Benchmark" in result
    assert_frame_equal(result["Benchmark"], ticker_data)
