from datetime import datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.cli.const import MARK_PRICE
from src.report.report import (
    create_best_and_worst_combined_page,
    create_best_and_worst_page,
    create_descriptions_page,
    create_metrics_page,
    create_new_trades_page,
    create_overlaps_page,
    create_title_page,
    create_top_holdings_page,
    get_aum,
    get_summary,
    plot_combined_pie_chart,
    plot_performance_charts,
    plot_pie_charts,
    plot_sector_weightings_page,
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
        "name": f"Name {ticker}",
        "description": f"Summary {ticker}",
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


def test_create_overlaps_page(mocker: Any) -> None:
    underlyings = pd.DataFrame(
        {
            "ticker": ["ETF1", "ETF2"],
            "Stock": ["AAPL", "GOOGL"],
            "Company": ["Apple Inc.", "Alphabet Inc."],
            "Weight": [0.5, 0.5],
        }
    )
    result_dict = {"Portfolio1": pd.DataFrame({"ticker": ["ETF1", "ETF2"]})}
    output_dir = "/tmp"

    result = create_overlaps_page(result_dict, underlyings, output_dir)

    assert result == ["/tmp/heatmap_Portfolio1.pdf"]


def test_plot_performance_charts(mocker: Any) -> None:
    mocker.patch("matplotlib.pyplot.savefig")
    plt = mocker.patch("matplotlib.pyplot.show")

    result_dict = {
        "Portfolio1": pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=3, freq="D"),
                "pnl_pct": [0.1, 0.2, 0.3],
                "portfolio_value": [100, 110, 120],
                "portfolio_pnl": [10, 20, 30],
                "total_cost": [1, 2, 3],
            }
        )
    }
    args = SimpleNamespace(
        start_date="2023-01-01", end_date="2023-01-03", timeframe="Daily"
    )

    result = plot_performance_charts(args, result_dict, "/tmp")
    assert result == "/tmp/performance.pdf"

    plot_performance_charts(args, result_dict)
    plt.assert_called_once()


def test_plot_combined_pie_chart(mocker: Any) -> None:
    mocker.patch("matplotlib.pyplot.savefig")

    result_dict = {
        "portfolio1": pd.DataFrame(
            {
                "date": [datetime(2023, 10, 1), datetime(2023, 10, 1)],
                "ticker": ["AAPL", "GOOGL"],
                "cumulative_quantity": [10, 15],
                "notional_value": [1500, 2000],
            }
        )
    }
    end_date = datetime(2023, 10, 1)
    other_threshold = 0.9
    output_dir = "/tmp"

    result = plot_combined_pie_chart(result_dict, end_date, other_threshold, output_dir)

    assert result == "/tmp/combined.pdf"


def test_create_title_page(mocker: Any) -> None:
    mocker.patch("fpdf.FPDF.output", return_value=None)
    mocker.patch("fpdf.FPDF.add_page", return_value=None)
    mocker.patch("fpdf.FPDF.set_font", return_value=None)
    mocker.patch("fpdf.FPDF.cell", return_value=None)
    mocker.patch("fpdf.FPDF.image", return_value=None)

    title = "Annual Report"
    aum = "1 Billion USD"
    image_file = "path/to/image.png"
    end_date = datetime(2023, 10, 1)
    output_dir = "/tmp"

    result = create_title_page(title, aum, image_file, end_date, output_dir)

    assert result == f"{output_dir}/title.pdf"


def test_plot_pie_charts(mocker: Any) -> None:
    plt = mocker.patch("matplotlib.pyplot.savefig")

    result_dict = {
        "Portfolio1": pd.DataFrame(
            {
                "date": [datetime(2023, 10, 1), datetime(2023, 10, 1)],
                "ticker": ["AAPL", "GOOGL"],
                "cumulative_quantity": [10, 15],
                "notional_value": [1500, 2500],
            }
        ),
        "Portfolio2": pd.DataFrame(
            {
                "date": [datetime(2023, 10, 1), datetime(2023, 10, 1)],
                "ticker": ["MSFT", "AMZN"],
                "cumulative_quantity": [5, 20],
                "notional_value": [1000, 3000],
            }
        ),
    }
    end_date = datetime(2023, 10, 1)
    other_threshold = 0.9
    output_dir = "/tmp"

    result = plot_pie_charts(result_dict, end_date, other_threshold, output_dir)

    assert result == "/tmp/weightings.pdf"
    plt.assert_called_once_with("/tmp/weightings.pdf")


def test_create_metrics_page(mocker: Any) -> None:
    metrics = pd.DataFrame(
        [
            {
                "Ticker": "ETF1",
                "Exp. Ratio": 0.01,
                "Div. Yield": 0.79,
                "Sharpe Ratio": 1.2,
                "Beta": 1.1,
                "PE Ratio": 25,
                "Volume": 100000,
                "Assets": 1000000,
                "YTD Return": 5.0,
                "3yr Return": 15.0,
            }
        ]
    )
    mocker.patch(
        "src.report.report.get_metrics",
        return_value=metrics,
    )
    mocker.patch(
        "src.report.report.df_to_pdf", return_value=["/path/to/pdf1", "/path/to/pdf2"]
    )

    result_dict = {
        "Portfolio1": pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-10-01")],
                "cumulative_quantity": [100],
                "ticker": ["ETF1"],
            }
        ),
        "Portfolio2": pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-10-01")],
                "cumulative_quantity": [200],
                "ticker": ["ETF2"],
            }
        ),
    }
    end_date = pd.Timestamp("2023-10-01")
    threshold = ["0.5"]
    operator = [">"]
    highlight = "red"
    output_dir = "/output/dir"

    result = create_metrics_page(
        result_dict, end_date, threshold, operator, highlight, output_dir
    )
    assert result == ["/path/to/pdf1", "/path/to/pdf2"]

    result = create_metrics_page(
        result_dict, end_date, [""], operator, highlight, output_dir
    )
    assert result == ["/path/to/pdf1", "/path/to/pdf2"]


def test_get_summary(mocker: Any) -> None:
    result_dict = {
        "ETF1": pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "portfolio_pnl": [1000, 1100],
                "portfolio_value": [10000, 11000],
                "quantity": [0, 10],
                "total_cost": [0, 100],
            }
        )
    }
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 2)
    timeframe = "YTD"
    comments = {"best": "Best performance", "worst": "Worst performance"}
    output_dir = "/tmp"

    result = get_summary(
        result_dict, start_date, end_date, timeframe, comments, output_dir
    )
    assert result == ["/tmp/summary_1.pdf"]

    result = get_summary(result_dict, start_date, end_date, timeframe, comments, "")
    assert result is None


def test_create_top_holdings_page(mocker: Any) -> None:
    mocker.patch("src.utils.pdf.df_to_pdf", return_value=["/path/to/pdf1.pdf"])

    result_dict = {
        "Tech": pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-10-01")],
                "cumulative_quantity": [100],
                "ticker": ["AAPL"],
                "notional_value": [1000.0],
            }
        )
    }
    underlyings = pd.DataFrame(
        {
            "ticker": ["AAPL", "GOOGL"],
            "Stock": ["AAPL", "GOOGL"],
            "Company": ["Apple Inc.", "Alphabet Inc."],
            "Weight": [50.0, 50.0],
        }
    )
    end_date = pd.Timestamp("2023-10-01")
    num_of_companies = 1
    threshold = 10.0
    output_dir = "/tmp"

    result = create_top_holdings_page(
        result_dict, underlyings, end_date, num_of_companies, threshold, output_dir
    )
    assert result == ["/tmp/top_holdings_1.pdf"]

    result = create_top_holdings_page(
        result_dict, underlyings, end_date, num_of_companies, 0.0, output_dir
    )
    assert result == ["/tmp/top_holdings_1.pdf"]


def test_plot_sector_weightings_page(mocker: Any) -> None:
    sectors = pd.DataFrame(
        [
            {
                "Ticker": ["VOO", "VOO", "VONG", "VONG"],
                "Sector": [
                    "Technology",
                    "Healthcare",
                    "Technology",
                    "Financial Services",
                ],
                "Weight": [60.0, 40.0, 70.0, 30.0],
            }
        ]
    )
    mocker.patch(
        "src.report.report.get_sector_weightings",
        return_value=sectors,
    )
    mocker.patch("matplotlib.pyplot.savefig")

    result_dict = {
        "portfolio1": pd.DataFrame(
            {
                "date": [datetime(2023, 10, 1), datetime(2023, 10, 1)],
                "ticker": ["VOO", "VONG"],
                "cumulative_quantity": [10, 15],
                "notional_value": [1500, 2000],
            }
        )
    }
    end_date = datetime(2023, 10, 1)
    output_dir = "/tmp"

    result = plot_sector_weightings_page(result_dict, end_date, output_dir)

    assert result == "/tmp/sectors.pdf"
