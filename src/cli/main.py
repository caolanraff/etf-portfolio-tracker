"""
ETF Portfolio Tracker.

This script tracks and analyses multiple ETF portfolios, given an Excel file with the trades made.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import argparse
import configparser
import logging
import math
import warnings
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cli.const import CHART_PALETTE
from src.report.calcs import (
    calculate_all_portfolio_pnl,
    calculate_sharpe_ratio,
    calculate_ytd,
)
from src.report.report import (
    create_best_and_worst_combined_page,
    create_best_and_worst_page,
    create_descriptions_page,
    create_new_trades_page,
    create_overlaps_page,
    create_title_page,
)
from src.utils.data import get_etf_underlyings, get_ticker_info, ticker_data
from src.utils.pdf import df_to_pdf, merge_pdfs, saved_pdf_files
from src.utils.types import DictFrame, Time

# Settings
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("seaborn-v0_8")


def parse_date(date_str: str, default_date: Time) -> Time:
    """Parse date string or return default date."""
    return datetime.strptime(date_str, "%Y-%m-%d") if date_str else default_date


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


def get_aum(result_dict: DictFrame, end_date: Time) -> str:
    """
    Calculate the Assets Under Management (AUM) based on the portfolio values in the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
        end_date: The end date of the report.
    Returns:
        The AUM value formatted as a string.
    """
    logging.info("Getting AUM")
    portfolio_val = 0

    for name, df in result_dict.items():
        if name == "Benchmark":
            continue
        res = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].iloc[0]
        portfolio_val += res["portfolio_value"]

    aum = f"${portfolio_val:,.0f}"
    return aum


def get_summary(args: Any, result_dict: DictFrame, save_to_file: bool) -> None:
    """
    Calculate and displays or saves the summary information based on the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
        save_to_file: Flag indicating whether to save the summary as a PDF file or print it.
    """
    logging.info("Getting summary information")
    val = []

    for key, df in result_dict.items():
        start = max(min(df["date"]).to_pydatetime(), args.start_date)
        first_day = df.loc[df["date"] == start].iloc[0]
        last_day = df.loc[df["date"] == args.end_date].iloc[0]
        trades = df.loc[(df["date"] > start) & (df["quantity"] != 0)]
        pnl = (
            (last_day["portfolio_pnl"] - first_day["portfolio_pnl"])
            / (first_day["portfolio_value"] + trades["total_cost"].sum())
        ) * 100
        val.append(pnl)

    summary = pd.DataFrame({"Portfolio": list(result_dict.keys()), args.timeframe: val})
    summary = summary.sort_values(by=args.timeframe, ascending=False)
    summary[args.timeframe] = summary[args.timeframe].round(3)
    summary = summary.reset_index(drop=True)
    summary["Notes"] = ""
    summary.loc[0, "Notes"] = config.get("SummaryPage", "best")
    summary.loc[summary.index[-1], "Notes"] = config.get("SummaryPage", "worst")

    if save_to_file:
        df_to_pdf("Summary", summary, f"{OUTPUT_DIR}/summary.pdf")
    else:
        print(summary)


def plot_performance_charts(
    args: Any, result_dict: DictFrame, save_to_file: bool
) -> None:
    """
    Plot performance charts based on the result dictionary and optionally save them to a file.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
        save_to_file: Flag indicating whether to save the performance charts as a PDF file or display them.
    """
    logging.info("Plotting performance charts")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.set_prop_cycle(color=CHART_PALETTE)
    ax2.set_prop_cycle(color=CHART_PALETTE)
    handles = []  # handles for the legend
    labels = []  # labels for the legend

    for name, df in result_dict.items():
        group = (
            df.groupby("date")
            .agg(
                {
                    "pnl_pct": "first",
                    "portfolio_value": "first",
                    "portfolio_pnl": "first",
                    "total_cost": "sum",
                }
            )
            .reset_index()
        )
        group.loc[0, "total_cost"] = 0.0
        group["pnl_pct_per_date"] = (
            (group["portfolio_pnl"] - group["portfolio_pnl"].iloc[0])
            / (group["portfolio_value"].iloc[0] + group["total_cost"].cumsum())
            * 100
        )

        (line1,) = ax1.plot(group["date"], group["pnl_pct"], label=name)
        if name not in labels:
            handles.append(line1)
            labels.append(name)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("PnL")
        ax1.set_title("Overall PnL Change", fontsize=12, fontweight="bold")
        ax1.set_xlim(args.start_date, args.end_date)

        (line2,) = ax2.plot(group["date"], group["pnl_pct_per_date"], label=name)
        if name not in labels:
            handles.append(line2)
            labels.append(name)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("PnL")
        ax2.set_title(f"{args.timeframe} PnL Change", fontsize=12, fontweight="bold")
        ax2.set_xlim(args.start_date, args.end_date)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=14)
    fig.legend(handles, labels)

    if save_to_file:
        file = f"{OUTPUT_DIR}/performance.pdf"
        plt.savefig(file)
        saved_pdf_files.append(file)
    else:
        plt.show()


def plot_pie_charts(result_dict: DictFrame, end_date: Time, output_dir: str) -> None:
    """
    Plot pie charts representing ETF weightings based on the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Plotting ETF weightings")
    n = len(result_dict)
    num_cols = 3
    num_rows = math.ceil(n / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
    if not isinstance(axs[0], np.ndarray):
        axs = [
            [axs[i * num_cols + j] for j in range(num_cols)] for i in range(num_rows)
        ]
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    for i, (key, df) in enumerate(result_dict.items()):
        row = i // num_cols
        col = i % num_cols
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]

        df = df.groupby("ticker")["notional_value"].sum()
        total_sum = df.sum()
        other = float(config.get("WeightingsPage", "other"))
        threshold = other * total_sum
        small_values = df[df < threshold]
        if len(small_values) > 1:
            df = df[df >= threshold]
            df["Other"] = small_values.sum()

        axs[row][col].set_prop_cycle(color=CHART_PALETTE)
        axs[row][col].pie(
            df.values,
            labels=df.index.to_list(),
            autopct="%1.1f%%",
            radius=1.2,
            textprops={"fontsize": 8},
        )
        axs[row][col].set_title(
            key, y=1.1, fontdict={"fontsize": 10, "fontweight": "bold"}
        )

    for i in range(n, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row][col])

    plt.suptitle("ETF Weightings", fontsize=12, fontweight="bold")
    file = f"{output_dir}/weightings.pdf"
    plt.savefig(file)
    saved_pdf_files.append(file)


def plot_combined_pie_chart(
    result_dict: DictFrame, end_date: Time, output_dir: str
) -> None:
    """
    Plot a combined pie chart representing the combined ETF weightings based on the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Plotting combined ETF weightings")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].copy()
        result_df = pd.concat([result_df, df], ignore_index=True)

    df = result_df.groupby("ticker")["notional_value"].sum()
    total_sum = df.sum()
    other = float(config.get("WeightingsPage", "other"))
    threshold = other * total_sum
    small_values = df[df < threshold]
    if len(small_values) > 1:
        df = df[df >= threshold]
        df["Other"] = small_values.sum()

    plt.clf()
    df.plot(
        kind="pie", autopct="%1.1f%%", colors=CHART_PALETTE, textprops={"fontsize": 8}
    )
    plt.title("Combined ETF Weightings", fontsize=12, fontweight="bold")
    plt.ylabel("")
    file = f"{output_dir}/combined.pdf"
    plt.savefig(file)
    saved_pdf_files.append(file)


def create_metrics_page(
    result_dict: DictFrame, end_date: Time, output_dir: str
) -> None:
    """
    Retrieve and process metrics for the ETFs in the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Getting metrics")

    df_list = []
    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        tickers = list(df["ticker"].unique())
        for ticker in tickers:
            dict = get_ticker_info(ticker)
            df = pd.DataFrame(
                [
                    {
                        "Portfolio": key,
                        "Ticker": ticker,
                        "Sharpe Ratio": calculate_sharpe_ratio(ticker, end_date),
                        "Beta": dict.get("beta3Year", None),
                        "Expense Ratio": None,
                        "PE Ratio": None,
                        "Yield": round(100 * dict.get("yield", 0.0), 2),
                        "YTD": calculate_ytd(ticker, end_date),
                    }
                ]
            )
            df_list.append(df)

    result_df = pd.concat(df_list, ignore_index=True)
    result_df = result_df.fillna("-")

    fields = [s for s in result_df.columns if s not in ["Portfolio", "Ticker"]]
    threshold = config.get("MetricsPage", "threshold").split(",")
    operator = config.get("MetricsPage", "operator").split(",")
    highlight = config.get("MetricsPage", "highlight")

    file = f"{output_dir}/metrics.pdf"
    if len(threshold) > 1:
        df_to_pdf(
            "Metrics",
            result_df,
            file,
            fields,
            [float(s) for s in threshold],
            operator,
            highlight,
        )
    else:
        df_to_pdf("Metrics", result_df, file)


def get_underlyings(result_dict: DictFrame) -> DictFrame:
    """
    Extract underlying stock information for all tickers in the result_dict.

    Args:
        result_dict: A dictionary containing the result data in the form of DataFrames, where the keys represent
        different categories and the values represent the corresponding DataFrames.
    Returns:
        A dictionary containing the extracted stock information for each category, where the keys represent the
        categories and the values represent the corresponding DataFrames with ticker, stock symbol, company name, and
        weight information.
    """
    logging.info("Downloading ETF underlyings")
    underlyings_dict = {}
    for key, df in result_dict.items():
        df = df.loc[(df["date"] == END_DATE) & (df["cumulative_quantity"] > 0)]
        tickers = df["ticker"].unique()
        underlyings = get_etf_underlyings(tickers)
        if underlyings.empty:
            continue
        underlyings_dict[key] = underlyings
    return underlyings_dict


def create_top_holdings_page(
    result_dict: DictFrame, underlyings_dict: DictFrame, output_dir: str
) -> None:
    """
    Retrieve the top holdings based on the provided result_dict and underlyings_dict.

    Args:
        result_dict: A dictionary containing the result data in the form of DataFrames, where the keys represent
        different categories and the values represent the corresponding DataFrames.
        underlyings_dict: A dictionary containing the extracted stock information for each category, where the
        keys represent the categories and the values represent the corresponding DataFrames with ticker, stock symbol,
        company name, and weight information.
    Returns:
        DataFrame containing the top holdings information, including portfolio, number of stocks,
        percentage of overlap, stock symbol, company name, and weight.
    """
    logging.info("Getting top holdings")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == END_DATE) & (df["cumulative_quantity"] > 0)]
        df = df[["ticker", "notional_value"]]
        underlyings = underlyings_dict[key]
        underlyings = underlyings.drop_duplicates(subset=["ticker", "Stock", "Company"])
        res = pd.merge(df, underlyings, on=["ticker"], how="left")
        res["Company"] = res["Company"].str.rstrip(".")
        res["symbol_notional"] = res["notional_value"] * (res["Weight"] / 100)
        grouped = res.groupby(["Stock", "Company"])["symbol_notional"].sum()
        grouped = grouped.reset_index()
        total_notional = df["notional_value"].sum()
        grouped["Weight"] = grouped["symbol_notional"] / total_notional * 100
        num_of_companies = config.get("HoldingsPage", "num_of_companies")
        holdings = grouped.sort_values("Weight", ascending=False).head(
            int(num_of_companies)
        )
        holdings["Portfolio"] = key
        stocks = underlyings["Stock"]
        holdings["No. of Stocks"] = len(stocks.unique())
        holdings["No. of Stocks"] = holdings["No. of Stocks"].apply(
            lambda x: "{:,.0f}".format(x)
        )
        holdings["% of Overlap"] = round(
            100 * (len(stocks) - len(stocks.unique())) / len(stocks), 2
        )
        holdings["Weight"] = [round(x, 2) for x in holdings["Weight"]]
        holdings = holdings[
            ["Portfolio", "No. of Stocks", "% of Overlap", "Stock", "Company", "Weight"]
        ]
        result_df = pd.concat([result_df, holdings], ignore_index=True)

    threshold = config.get("HoldingsPage", "threshold")

    file = f"{output_dir}/holdings.pdf"
    if len(threshold) > 0:
        df_to_pdf(
            "Top Holdings",
            result_df,
            file,
            ["Weight"],
            [float(threshold)],
            [">"],
            "red",
        )
    else:
        df_to_pdf("Top Holdings", result_df, file)


def summary(args: Any, config: Any) -> None:
    """Run a summary report, printing the outputs.

    Run a report for a specific timeframe, calculates portfolio P&L, retrieves AUM (Assets Under Management),
    generates a summary, and plots performance charts.
    """
    logging.info(
        f"Running report for {args.timeframe} ({args.start_date:%Y-%m-%d} - {args.end_date:%Y-%m-%d})"
    )

    filename = config.get("Input", "file")
    file = f"{args.path}/data/input/{filename}"
    benchmark = config.get("Input", "benchmark")
    res_dict = calculate_all_portfolio_pnl(
        file, args.start_date, args.end_date, benchmark
    )

    aum = get_aum(res_dict, args.end_date)
    logging.info(f"AUM: {aum}")

    get_summary(args, res_dict, False)
    plot_performance_charts(args, res_dict, False)


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

    filename = config.get("Input", "file")
    file = f"{args.path}/data/input/{filename}"
    benchmark = config.get("Input", "benchmark")
    res_dict = calculate_all_portfolio_pnl(
        file, args.start_date, args.end_date, benchmark
    )

    aum = get_aum(res_dict, args.end_date)
    title = config.get("TitlePage", "title")
    image = config.get("TitlePage", "image")
    file = create_title_page(
        title, aum, image, args.end_date, f"{args.path}/data/output"
    )
    saved_pdf_files.append(file)

    get_summary(args, res_dict, True)
    plot_performance_charts(args, res_dict, True)

    res_dict.pop("Benchmark", None)
    create_new_trades_page(res_dict, f"{args.path}/data/output")
    create_best_and_worst_page(res_dict, args.end_date, f"{args.path}/data/output")
    create_best_and_worst_combined_page(
        res_dict,
        ticker_data,
        args.start_date,
        args.end_date,
        f"{args.path}/data/output",
    )
    plot_pie_charts(res_dict, args.end_date, f"{args.path}/data/output")
    plot_combined_pie_chart(res_dict, args.end_date, f"{args.path}/data/output")
    create_metrics_page(res_dict, args.end_date, f"{args.path}/data/output")

    under_dict = get_underlyings(res_dict)
    if len(under_dict) > 1:
        create_top_holdings_page(res_dict, under_dict, f"{args.path}/data/output")
        file_list = create_overlaps_page(
            res_dict, under_dict, f"{args.path}/data/output"
        )
        saved_pdf_files.extend(file_list)

    create_descriptions_page(sorted(ticker_data.keys()), f"{args.path}/data/output")
    filename = config.get("Output", "file")
    merge_pdfs(saved_pdf_files, f"{OUTPUT_DIR}/{filename}")
    logging.info("Complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    args = parse_arguments()
    START_DATE = args.start_date
    END_DATE = args.end_date
    OUTPUT_DIR = f"{args.path}/data/output"

    config = configparser.ConfigParser()
    config.read(f"{args.path}/{args.config}")

    if args.report:
        report(args, config)
    else:
        summary(args, config)
