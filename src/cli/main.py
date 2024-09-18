"""
ETF Portfolio Tracker.

This script tracks and analyses multiple ETF portfolios, given an Excel file with the trades made.

Author: Caolan Rafferty
Date: 2023-07-02
"""
import argparse
import configparser
import logging
from datetime import datetime, timedelta
from typing import Any

from src.report.calcs import calculate_all_portfolio_pnl
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
)
from src.utils.data import ticker_data
from src.utils.pdf import merge_pdfs, saved_pdf_files
from src.utils.util import parse_date


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeframe", default="MTD", type=str, help="timeframe [MTD|YTD|adhoc]"
    )
    parser.add_argument("--start", default=None, type=str, help="start date")
    parser.add_argument("--end", default=None, type=str, help="end date")
    parser.add_argument("--report", action="store_true", help="generate PDF report")
    parser.add_argument("--path", default="./", type=str, help="directory path")
    parser.add_argument(
        "--config", default="config/default.ini", type=str, help="config file"
    )
    args = parser.parse_args()

    now = datetime.now()
    timeframe_mapping = {
        "MTD": datetime(now.year, now.month, 1),
        "YTD": datetime(now.year, 1, 1),
    }

    args.start_date = parse_date(args.start, timeframe_mapping.get(args.timeframe))
    if args.start_date is None:
        raise ValueError("Unknown timeframe: {}".format(args.timeframe))

    args.start_date += timedelta(days=-1)
    args.end_date = parse_date(args.end, datetime(now.year, now.month, now.day))

    return args


def summary(args: Any, config: Any) -> None:
    """Run a summary report, printing the outputs.

    Run a report for a specific timeframe, calculates portfolio P&L, retrieves AUM (Assets Under Management),
    generates a summary, and plots performance charts.
    """
    logging.info(
        f"Running report for {args.timeframe} ({args.start_date:%Y-%m-%d} - {args.end_date:%Y-%m-%d})"
    )

    logging.info("Calculating portfolio PnLs")
    filename = config.get("Input", "file")
    file = f"{args.path}/data/input/{filename}"
    benchmark = config.get("Input", "benchmark")
    res_dict = calculate_all_portfolio_pnl(
        file, args.start_date, args.end_date, benchmark
    )

    logging.info("Getting AUM")
    aum = get_aum(res_dict, args.end_date)
    logging.info(f"AUM: {aum}")

    logging.info("Getting summary information")
    comments = {
        "best": config.get("SummaryPage", "best"),
        "worst": config.get("SummaryPage", "worst"),
    }
    get_summary(res_dict, args.start_date, args.end_date, args.timeframe, comments)
    plot_performance_charts(args, res_dict)

    logging.info("Complete")


def report(args: Any, config: Any) -> None:
    """Run a comprehensive report, saving to PDF.

    Run a comprehensive report for a specific timeframe, including portfolio P&L calculations, AUM retrieval,
    title page creation, summary generation, performance chart plotting, new trades analysis, exclusion of specific
    portfolios, best and worst performers analysis, pie chart plotting, combined pie chart plotting, metrics calculation,
    extraction of underlyings information, top holdings retrieval, ETF overlap analysis, and merging of PDF files.
    """
    logging.info(
        f"Running report for {args.timeframe} ({args.start_date:%Y-%m-%d} - {args.end_date:%Y-%m-%d})"
    )

    logging.info("Calculating portfolio PnLs")
    filename = config.get("Input", "file")
    file = f"{args.path}/data/input/{filename}"
    benchmark = config.get("Input", "benchmark")
    res_dict = calculate_all_portfolio_pnl(
        file, args.start_date, args.end_date, benchmark
    )

    logging.info("Getting AUM")
    aum = get_aum(res_dict, args.end_date)

    logging.info("Creating title page")
    title = config.get("TitlePage", "title")
    image = config.get("TitlePage", "image")
    title_page = create_title_page(
        title, aum, image, args.end_date, f"{args.path}/data/output"
    )
    saved_pdf_files.append(title_page)

    logging.info("Getting summary information")
    comments = {
        "best": config.get("SummaryPage", "best"),
        "worst": config.get("SummaryPage", "worst"),
    }
    summary = get_summary(
        res_dict,
        args.start_date,
        args.end_date,
        args.timeframe,
        comments,
        f"{args.path}/data/output",
    )
    saved_pdf_files.extend(summary)

    logging.info("Plotting performance charts")
    perf_charts = plot_performance_charts(args, res_dict, f"{args.path}/data/output")
    saved_pdf_files.append(perf_charts)
    # Don't need the brenchmark for the rest of the analysis
    res_dict.pop("Benchmark", None)

    logging.info("Creating new trades page")
    new_trades = create_new_trades_page(res_dict, f"{args.path}/data/output")
    saved_pdf_files.extend(new_trades)

    logging.info("Getting best and worst ETFs page")
    best_and_worst = create_best_and_worst_page(
        res_dict, args.end_date, f"{args.path}/data/output"
    )
    saved_pdf_files.extend(best_and_worst)

    logging.info("Getting combined best and worst ETFs page")
    best_and_worst_comb = create_best_and_worst_combined_page(
        res_dict,
        ticker_data,
        args.start_date,
        args.end_date,
        f"{args.path}/data/output",
    )
    saved_pdf_files.extend(best_and_worst_comb)

    logging.info("Plotting ETF weightings")
    threshold = config.get("WeightingsPage", "other")
    threshold = float(threshold) if len(threshold) > 0 else 0.0
    pie_charts = plot_pie_charts(
        res_dict, args.end_date, threshold, f"{args.path}/data/output"
    )
    saved_pdf_files.append(pie_charts)

    logging.info("Plotting combined ETF weightings")
    combined_pie_charts = plot_combined_pie_chart(
        res_dict, args.end_date, threshold, f"{args.path}/data/output"
    )
    saved_pdf_files.append(combined_pie_charts)

    logging.info("Getting metrics")
    threshold = config.get("MetricsPage", "threshold").split(",")
    operator = config.get("MetricsPage", "operator").split(",")
    highlight = config.get("MetricsPage", "highlight")
    metrics = create_metrics_page(
        res_dict,
        args.end_date,
        threshold,
        operator,
        highlight,
        f"{args.path}/data/output",
    )
    saved_pdf_files.extend(metrics)

    logging.info("Getting top holdings")
    num_of_companies = config.get("HoldingsPage", "num_of_companies")
    num_of_companies = int(num_of_companies) if len(num_of_companies) > 0 else 1
    threshold = config.get("HoldingsPage", "threshold")
    threshold = float(threshold) if len(threshold) > 0 else 0.0
    holdings = create_top_holdings_page(
        res_dict, args.end_date, num_of_companies, threshold, f"{args.path}/data/output"
    )
    saved_pdf_files.extend(holdings)

    logging.info("Plotting ETF overlap heatmap")
    overlaps = create_overlaps_page(res_dict, f"{args.path}/data/output")
    saved_pdf_files.extend(overlaps)

    logging.info("Creating description page")
    create_descriptions_page(sorted(ticker_data.keys()), f"{args.path}/data/output")

    output_file = config.get("Output", "file")
    merge_pdfs(saved_pdf_files, f"{args.path}/data/output/{output_file}")

    logging.info("Complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments()
    config = configparser.ConfigParser()
    config.read(args.config)

    report(args, config) if args.report else summary(args, config)
