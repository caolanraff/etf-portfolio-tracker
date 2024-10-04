from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.cli.const import MARK_PRICE
from src.report.report import (
    create_best_and_worst_combined_page,
    create_best_and_worst_page,
    create_descriptions_page,
    create_new_trades_page,
    get_aum,
)


def test_get_aum() -> None:
    result_dict = {
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
        "Benchmark": pd.DataFrame(
            {
                "date": [datetime(2023, 1, 1)],
                "cumulative_quantity": [5],
                "portfolio_value": [500],
            }
        ),
    }
    end_date = datetime(2023, 1, 2)

    result = get_aum(result_dict, end_date)
    expected = "$3,500"

    assert result == expected


def test_create_new_trades_page(mocker: Any) -> None:
    mock_df_to_pdf = mocker.patch(
        "src.report.report.df_to_pdf", return_value=["/path/to/pdf1.pdf"]
    )

    result_dict = {
        "Portfolio1": pd.DataFrame(
            {"ticker": ["AAPL", "GOOGL", "MSFT"], "quantity": [10, -5, 15]}
        ),
        "Portfolio2": pd.DataFrame({"ticker": ["TSLA", "AMZN"], "quantity": [-10, 20]}),
    }
    output_dir = "/output/dir"
    result = create_new_trades_page(result_dict, output_dir)

    assert result == ["/path/to/pdf1.pdf"]
    mock_df_to_pdf.assert_called_once()


def test_create_best_and_worst_page(mocker: Any) -> None:
    mocker.patch("src.report.report.df_to_pdf", return_value=["/tmp/best_worst.pdf"])

    result_dict = {
        "Portfolio1": pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "date": [np.datetime64("2023-01-01"), np.datetime64("2023-01-01")],
                "cumulative_quantity": [10, 20],
                "notional_value": [1000, 2000],
                "total_pnl": [100, 200],
                "market_price": [10, 20],
            }
        )
    }
    end_date = np.datetime64("2023-01-01")
    output_dir = "/tmp"

    result = create_best_and_worst_page(result_dict, end_date, output_dir)

    assert result == ["/tmp/best_worst.pdf"]


def test_create_best_and_worst_combined_page(mocker: Any) -> None:
    mocker.patch("src.report.report.df_to_pdf", return_value=["/tmp/mock.pdf"])

    ticker_data = {
        "ETF1": pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                MARK_PRICE: [100, 105, 110, 115, 120],
            }
        ).set_index("date"),
        "ETF2": pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                MARK_PRICE: [200, 190, 180, 170, 160],
            }
        ).set_index("date"),
    }
    result_dict = {
        "result1": pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "ticker": ["ETF1", "ETF2", "ETF1", "ETF2", "ETF1"],
                "cumulative_quantity": [10, 20, 30, 40, 50],
            }
        )
    }
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 5)
    output_dir = "/tmp"

    result = create_best_and_worst_combined_page(
        result_dict, ticker_data, start_date, end_date, output_dir
    )

    assert result == ["/tmp/mock.pdf"]


def test_create_descriptions_page(mocker: Any) -> None:
    mock_get_ticker_info = mocker.patch("src.report.report.get_ticker_info")
    mock_save_paragraphs_to_pdf = mocker.patch(
        "src.report.report.save_paragraphs_to_pdf"
    )

    mock_get_ticker_info.side_effect = lambda ticker: {
        "shortName": f"Name {ticker}",
        "longBusinessSummary": f"Summary {ticker}",
    }
    mock_save_paragraphs_to_pdf.return_value = "/fake/dir/etf_descriptions.pdf"

    tickers = ["AAPL", "GOOGL"]
    output_dir = "/fake/dir"
    result = create_descriptions_page(tickers, output_dir)

    assert result == "/fake/dir/etf_descriptions.pdf"
    mock_save_paragraphs_to_pdf.assert_called_once_with(
        "ETF Descriptions",
        ["Name AAPL (AAPL)", "Name GOOGL (GOOGL)"],
        ["Summary AAPL", "Summary GOOGL"],
        output_dir,
    )
