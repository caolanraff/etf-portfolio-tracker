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
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs
import seaborn as sns
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

import utils.pdf as pdf
from utils.data import (
    get_etf_underlyings,
    get_ticker_data,
    get_ticker_info,
    ticker_data,
)
from utils.pdf import saved_pdf_files

# Variable definitions
DictFrame = Dict[str, pd.DataFrame]
MARK_PRICE = "Adj Close"

# Settings
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("seaborn-v0_8")
palette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


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

    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d")
        if args.start
        else timeframe_mapping.get(args.timeframe)
    )
    if start_date is None:
        raise ValueError("Unknown timeframe: {}".format(args.timeframe))
    args.start_date = start_date + timedelta(days=-1)

    args.end_date = (
        datetime.strptime(args.end, "%Y-%m-%d")
        if args.end
        else datetime(now.year, now.month, now.day)
    )

    return args


def calculate_entry_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average entry price.

    This is the weighted average price, for buys only.
    """
    res = df[df["quantity"] > 0].copy()
    res["cumulative_quantity"] = res["quantity"].cumsum()
    res["cumulative_weighted_price"] = (res["quantity"] * res["price"]).cumsum()
    res["average_entry_price"] = (
        res["cumulative_weighted_price"] / res["cumulative_quantity"]
    )
    res = res[["date", "average_entry_price"]]
    return res


def calculate_portfolio_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the profit and loss (PnL) for a specific portfolio.

    Args:
        df: A dataframe of the portfolio executions data.
    Returns:
        A DataFrame containing the calculated PnL for the portfolio.
    """
    df["date"] = pd.to_datetime(df["date"])
    # Aggregate the trades by date, ticker
    df = (
        df.groupby(["date", "ticker"])
        .agg(
            {
                "quantity": "sum",
                "price": lambda x: np.average(x, weights=df.loc[x.index, "quantity"]),
            }
        )
        .reset_index()
    )
    max_date = pd.Timestamp(end_date)
    df = df[df["date"] < max_date]
    grouped = df.groupby("ticker")
    result_df = pd.DataFrame()

    # Calculate running cumulative quantity and running average entry price per ticker
    for name, group in grouped:
        # Create a date range from the earliest to latest date for this ticker
        date_range = pd.date_range(start=group["date"].min(), end=max_date, freq="D")
        # Create a new dataframe with the complete date range and the ticker symbol
        date_df = pd.DataFrame({"date": date_range, "ticker": name})
        # Merge the original dataframe with the new dataframe
        merged_df = pd.merge(date_df, group, on=["date", "ticker"], how="outer")
        # Fill missing values with 0
        merged_df = merged_df.fillna(0)
        # Calculate running cumulative quantity
        merged_df["cumulative_quantity"] = merged_df["quantity"].cumsum()
        # Calculate cumulative cost (buys)
        merged_df["total_cost"] = np.where(
            merged_df["quantity"] > 0, merged_df["quantity"] * merged_df["price"], 0
        )
        merged_df["cumulative_cost"] = merged_df["total_cost"].cumsum()
        # Calculate cumulative proceeds (sells)
        merged_df["total_proceeds"] = np.where(
            merged_df["quantity"] < 0, merged_df["quantity"] * merged_df["price"], 0
        )
        merged_df["cumulative_proceeds"] = merged_df["total_proceeds"].cumsum()
        # Calculate average entry price
        entry_price = calculate_entry_price(merged_df)
        merged_df = pd.merge(merged_df, entry_price, on="date", how="left")
        merged_df = merged_df.groupby("ticker").apply(
            lambda x: x.fillna(method="ffill")
        )
        # merged_df["cumulative_cost"] = merged_df["cumulative_quantity"] * merged_df["average_entry_price"]
        result_df = pd.concat([result_df, merged_df], ignore_index=True)

    # Download the adjusted close price from Yahoo Finance for each ticker and date
    min_date = result_df["date"].min()
    min_date = min_date - timedelta(days=7)
    prices = {}
    for ticker in result_df["ticker"].unique():
        data = get_ticker_data(ticker)
        data = data.loc[min_date:max_date][MARK_PRICE]
        prices[ticker] = data

    # Merge the price data onto the original dataframe
    result_df["market_price"] = result_df.apply(
        lambda row: prices[row["ticker"]].loc[pd.Timestamp(row["date"])], axis=1
    )
    # Calculate the notional value based on the market price
    result_df["notional_value"] = (
        result_df["cumulative_quantity"] * result_df["market_price"]
    )
    # Calculate PnL
    result_df["unrealised_pnl"] = result_df["cumulative_quantity"] * (
        result_df["market_price"] - result_df["average_entry_price"]
    )
    result_df["realised_pnl"] = np.where(
        result_df["quantity"] < 0,
        abs(result_df["quantity"])
        * (result_df["price"] - result_df["average_entry_price"]),
        0,
    )
    result_df["realised_pnl"] = result_df.groupby("ticker")["realised_pnl"].cumsum()
    result_df["total_pnl"] = result_df["unrealised_pnl"] + result_df["realised_pnl"]
    # Sort the dataframe by date
    result_df = result_df.reset_index()
    result_df = result_df.sort_values(["date", "index"])
    result_df = result_df.drop("index", axis=1)
    # Calculate the PNL per date
    result_df["portfolio_pnl"] = result_df.groupby("date")["total_pnl"].transform("sum")
    # Calculate the portfolio cost (total_cost - total_proceeds)
    result_df["portfolio_cost"] = result_df["cumulative_cost"] + abs(
        result_df["cumulative_proceeds"]
    )
    result_df["portfolio_cost"] = result_df.groupby("date")["portfolio_cost"].transform(
        "sum"
    )
    # Calculate the portfolio value per date
    result_df["portfolio_value"] = result_df.groupby("date")[
        "notional_value"
    ].transform("sum")
    # Calculate the PNL percentage
    result_df["pnl_pct"] = 100 * (
        result_df["portfolio_pnl"] / result_df["portfolio_cost"]
    )

    result_df = result_df.reset_index(drop=True)
    return result_df


def calculate_all_portfolio_pnl() -> DictFrame:
    """
    Calculate the profit and loss (PnL) for all portfolios.

    Returns:
        A dictionary containing the calculated PnL for each portfolio.
    """
    logging.info("Calculating portfolio PnLs")
    result_dict = {}
    file = config.get("Input", "file")
    path = f"{args.path}/data/input/{file}"
    sheets = pd.ExcelFile(path).sheet_names
    for sheet in sheets:
        data = pd.read_excel(path, sheet_name=sheet)
        if len(data) == 0:
            logging.warning(f"Tab is empty for {sheet}")
            continue
        res = calculate_portfolio_pnl(data)
        res = res[(res["date"] >= start_date) & (res["date"] <= end_date)]
        group = res.groupby("ticker")["cumulative_quantity"].sum()
        tickers = group[group == 0].index
        res = res[~res["ticker"].isin(tickers)]
        result_dict[sheet] = res

    # add benchmark portfolio
    bticker = config.get("Input", "benchmark")
    if bticker != "":
        data = get_ticker_data(bticker)
        data = data.loc[start_date:end_date][MARK_PRICE]
        benchmark = pd.DataFrame(
            {"date": data.index, "ticker": len(data) * [bticker], "price": data.values}
        )
        benchmark = benchmark[benchmark["date"].dt.is_month_end]
        benchmark["quantity"] = 1.0
        result_dict["Benchmark"] = calculate_portfolio_pnl(benchmark)

    return result_dict


def create_title_page(aum: str) -> None:
    """
    Create a title page for a PDF document with specified information.

    Args:
        aum: Assets Under Management (AUM) value to be displayed on the title page.
    """
    logging.info("Creating title page")
    pdf_output = FPDF()
    pdf_output.add_page()
    title = config.get("TitlePage", "title")
    subtitle = f"{end_date.strftime('%B %Y')} Meeting"
    aum = f"AUM: {aum}"
    pdf_output.set_font("Arial", "B", 36)
    pdf_output.cell(0, 80, title, 0, 1, "C")
    pdf_output.set_font("Arial", "", 24)
    pdf_output.cell(0, 20, subtitle, 0, 1, "C")
    pdf_output.set_font("Arial", "", 16)
    pdf_output.cell(0, 20, aum, 0, 1, "C")
    image = config.get("TitlePage", "image")
    if image != "":
        image_file = f"{args.path}/data/input/{image}"
        pdf_output.image(image_file, x=55, y=150, w=100, h=100)
    file = f"{output_dir}/title.pdf"
    pdf_output.output(file)
    saved_pdf_files.append(file)


def get_aum(result_dict: DictFrame) -> str:
    """
    Calculate the Assets Under Management (AUM) based on the portfolio values in the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
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


def get_summary(result_dict: DictFrame, save_to_file: bool) -> None:
    """
    Calculate and displays or saves the summary information based on the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
        save_to_file: Flag indicating whether to save the summary as a PDF file or print it.
    """
    logging.info("Getting summary information")
    val = []

    for key, df in result_dict.items():
        start = max(min(df["date"]).to_pydatetime(), start_date)
        first_day = df.loc[df["date"] == start].iloc[0]
        last_day = df.loc[df["date"] == end_date].iloc[0]
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
        pdf.df_to_pdf("Summary", summary, f"{output_dir}/summary.pdf")
    else:
        print(summary)


def plot_performance_charts(result_dict: DictFrame, save_to_file: bool) -> None:
    """
    Plot performance charts based on the result dictionary and optionally save them to a file.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
        save_to_file: Flag indicating whether to save the performance charts as a PDF file or display them.
    """
    logging.info("Plotting performance charts")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.set_prop_cycle(color=palette)
    ax2.set_prop_cycle(color=palette)
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
        ax1.set_xlim(start_date, end_date)

        (line2,) = ax2.plot(group["date"], group["pnl_pct_per_date"], label=name)
        if name not in labels:
            handles.append(line2)
            labels.append(name)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("PnL")
        ax2.set_title(f"{args.timeframe} PnL Change", fontsize=12, fontweight="bold")
        ax2.set_xlim(start_date, end_date)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=14)
    fig.legend(handles, labels)

    if save_to_file:
        file = f"{output_dir}/performance.pdf"
        plt.savefig(file)
        saved_pdf_files.append(file)
    else:
        plt.show()


def create_new_trades_page(result_dict: DictFrame) -> None:
    """
    Retrieve the new trades from the result dictionary and saves them as a PDF report.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Getting new trades")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        buys = df.loc[df["quantity"] > 0]["ticker"].to_list()
        sells = df.loc[df["quantity"] < 0]["ticker"].to_list()
        df = pd.DataFrame(
            {
                "Portfolio": [key],
                "Buys": [", ".join(set(buys))],
                "Sells": [", ".join(set(sells))],
            }
        )
        result_df = pd.concat([result_df, df], ignore_index=True)

    pdf.df_to_pdf("New Trades", result_df, f"{output_dir}/new_trades.pdf")


def create_best_and_worst_page(result_dict: DictFrame) -> None:
    """
    Compute the best and worst performers among the ETFs in the result dictionary and saves the results as a PDF report.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Getting best and worst ETFs")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        first_day = df.groupby("ticker").first()
        last_day = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        merged_df = pd.merge(
            first_day, last_day, on="ticker", suffixes=("_start", "_end")
        )
        total_notional = last_day["notional_value"].sum()

        merged_df["pnl_pct"] = (
            (merged_df["total_pnl_end"] - merged_df["total_pnl_start"])
            / total_notional
            * 100
        )
        merged_df["price_pct"] = (
            (merged_df["market_price_end"] - merged_df["market_price_start"])
            / abs(merged_df["market_price_start"])
            * 100
        )

        best_price_pct = merged_df.loc[merged_df["price_pct"].idxmax(), "ticker"]
        max_price_pct = round(merged_df["price_pct"].max(), 2)
        best_price_pct = f"{best_price_pct} ({max_price_pct}%)"

        worst_price_pct = merged_df.loc[merged_df["price_pct"].idxmin(), "ticker"]
        min_price_pct = round(merged_df["price_pct"].min(), 2)
        worst_price_pct = f"{worst_price_pct} ({min_price_pct}%)"

        best_pnl_pct = merged_df.loc[merged_df["pnl_pct"].idxmax(), "ticker"]
        max_pnl_pct = round(merged_df["pnl_pct"].max(), 2)
        best_pnl_pct = f"{best_pnl_pct} ({max_pnl_pct}%)"

        worst_pnl_pct = merged_df.loc[merged_df["pnl_pct"].idxmin(), "ticker"]
        min_pnl_pct = round(merged_df["pnl_pct"].min(), 2)
        worst_pnl_pct = f"{worst_pnl_pct} ({min_pnl_pct}%)"

        summary = pd.DataFrame(
            {
                "Portfolio": [key],
                "Best Price Pct": [best_price_pct],
                "Worst Price Pct": [worst_price_pct],
                "Best PnL Pct": [best_pnl_pct],
                "Worst PnL Pct": [worst_pnl_pct],
            }
        )

        result_df = pd.concat([result_df, summary], ignore_index=True)

    pdf.df_to_pdf(
        "Best & Worst Performers", result_df, f"{output_dir}/best_and_worst.pdf"
    )


def create_best_and_worst_combined_page(result_dict: DictFrame) -> None:
    """
    Combine the best and worst performing ETFs based on their returns.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Getting combined best and worst ETFs")
    returns = pd.DataFrame(columns=["Ticker", "Returns"])
    result_df = pd.DataFrame()
    tickers = []

    for key, df in ticker_data.items():
        df = df.loc[start_date:end_date]
        first_close = df[MARK_PRICE].iloc[0]
        last_close = df[MARK_PRICE].iloc[-1]
        percentage_change = (last_close - first_close) / first_close * 100
        df = pd.DataFrame({"Ticker": [key], "Returns": [round(percentage_change, 2)]})
        returns = pd.concat([returns, df], ignore_index=True)

    for key, df in result_dict.items():
        start = max(min(df["date"]).to_pydatetime(), start_date)
        df = df.loc[(df["date"] == start) & (df["cumulative_quantity"] > 0)]
        tickers += df["ticker"].to_list()

    returns = returns.loc[returns["Ticker"].isin(tickers)]
    top = returns.sort_values(by="Returns", ascending=False)
    top = top.head(5)
    top.reset_index(drop=True, inplace=True)
    result_df["Top 5 ETFs"] = top.apply(
        lambda row: f"{row['Ticker']} ({row['Returns']}%)", axis=1
    )
    bottom = returns.sort_values(by="Returns", ascending=True)
    bottom = bottom.head(5)
    bottom.reset_index(drop=True, inplace=True)
    result_df["Bottom 5 ETFs"] = bottom.apply(
        lambda row: f"{row['Ticker']} ({row['Returns']}%)", axis=1
    )

    pdf.df_to_pdf(
        "Best & Worst Performers Combined",
        result_df,
        f"{output_dir}/best_and_worst_combined.pdf",
    )


def plot_pie_charts(result_dict: DictFrame) -> None:
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

        axs[row][col].set_prop_cycle(color=palette)
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


def plot_combined_pie_chart(result_dict: DictFrame) -> None:
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
    df.plot(kind="pie", autopct="%1.1f%%", colors=palette, textprops={"fontsize": 8})
    plt.title("Combined ETF Weightings", fontsize=12, fontweight="bold")
    plt.ylabel("")
    file = f"{output_dir}/combined.pdf"
    plt.savefig(file)
    saved_pdf_files.append(file)


def calculate_sharpe_ratio(ticker: str) -> float:
    """
    Calculate the Sharpe ratio for a given ETF ticker.

    Args:
        ticker: The ETF ticker symbol.
    Returns:
        The Sharpe ratio.
    """
    data = get_ticker_data(ticker)
    min_date = end_date - timedelta(days=5 * 365)
    data = data.loc[min_date:end_date]
    pct_chg = data[MARK_PRICE].pct_change()
    sharpe = qs.stats.sharpe(pct_chg).round(2)
    return float(sharpe)


def calculate_ytd(ticker: str) -> Any:
    """
    Calculate the YTD for a given ETF ticker.

    Args:
        ticker: The ETF ticker symbol.
    Returns:
        YTD.
    """
    data = get_ticker_data(ticker)
    min_date = pd.to_datetime(end_date.year, format="%Y")
    data = data.loc[min_date:end_date]
    start = data.head(1)[MARK_PRICE][0]
    end = data.tail(1)[MARK_PRICE][0]
    ytd = ((end - start) / start) * 100
    return round(ytd, 2)


def create_metrics_page(result_dict: DictFrame) -> None:
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
                        "Sharpe Ratio": calculate_sharpe_ratio(ticker),
                        "Beta": dict.get("beta3Year", None),
                        "Expense Ratio": None,
                        "PE Ratio": None,
                        "Yield": round(100 * dict.get("yield", 0.0), 2),
                        "YTD": calculate_ytd(ticker),
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
        pdf.df_to_pdf(
            "Metrics",
            result_df,
            file,
            fields,
            [float(s) for s in threshold],
            operator,
            highlight,
        )
    else:
        pdf.df_to_pdf("Metrics", result_df, file)


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
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        tickers = df["ticker"].unique()
        underlyings = get_etf_underlyings(tickers)
        if underlyings.empty:
            continue
        underlyings_dict[key] = underlyings
    return underlyings_dict


def create_top_holdings_page(
    result_dict: DictFrame, underlyings_dict: DictFrame
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
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
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
        pdf.df_to_pdf(
            "Top Holdings",
            result_df,
            file,
            ["Weight"],
            [float(threshold)],
            [">"],
            "red",
        )
    else:
        pdf.df_to_pdf("Top Holdings", result_df, file)


def create_overlaps_page(result_dict: DictFrame, underlyings_dict: DictFrame) -> None:
    """
    Generate an ETF overlap heatmap based on the provided result_dict and underlyings_dict.

    Args:
        result_dict: A dictionary containing the result data in the form of DataFrames, where the keys represent
        different categories and the values represent the corresponding DataFrames.
        underlyings_dict: A dictionary containing the extracted stock information for each category, where the
        keys represent the categories and the values represent the corresponding DataFrames with ticker, stock symbol,
        company name, and weight information.
    """
    logging.info("Plotting ETF overlap heatmap")

    for key, df in result_dict.items():
        underlyings = underlyings_dict[key]
        underlyings["Stock"] = (
            underlyings["Stock"].replace("N/A", np.nan).fillna(underlyings["Company"])
        )
        group = underlyings.groupby("ticker")["Stock"].apply(list)
        overlaps = pd.DataFrame(columns=["ETF1", "ETF2", "Overlap"])

        for k in group.keys():
            for i in group.keys():
                val = (
                    100
                    * len(set(group[k]).intersection(set(group[i])))
                    / len(set(group[k]))
                )
                df = pd.DataFrame({"ETF1": [k], "ETF2": [i], "Overlap": [val]})
                overlaps = pd.concat([overlaps, df], ignore_index=True)

        matrix = overlaps.pivot(index="ETF1", columns="ETF2", values="Overlap")
        matrix.index.name = None
        matrix.columns.name = None

        plt.clf()
        sns_plot = sns.heatmap(
            matrix, cmap="Blues", annot=True, fmt=".2f", annot_kws={"fontsize": 8}
        )
        sns_plot.figure.set_size_inches(10, 7)
        sns_plot.set_title(f"ETF Overlaps - {key}", fontsize=12, fontweight="bold")
        file = f"{output_dir}/heatmap_{key}.pdf"
        pp = PdfPages(file)
        pp.savefig(sns_plot.figure)
        pp.close()
        saved_pdf_files.append(file)


def create_descriptions_page() -> None:
    """Create ETF descriptions page."""
    logging.info("Creating description page")

    tickers = sorted(ticker_data.keys())
    headers = []
    paragraphs = []

    for i in tickers:
        data = get_ticker_info(i)
        if "longBusinessSummary" in data:
            name = data["shortName"]
            headers += [f"{name} ({i})"]
            paragraphs += [data["longBusinessSummary"]]

    pdf.save_paragraphs_to_pdf(
        "ETF Descriptions", headers, paragraphs, f"{output_dir}/descriptions.pdf"
    )


def summary() -> None:
    """Run a summary report, printing the outputs.

    Run a report for a specific timeframe, calculates portfolio P&L, retrieves AUM (Assets Under Management),
    generates a summary, and plots performance charts.
    """
    logging.info(
        f"Running report for {args.timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    logging.info(f"AUM: {aum}")
    get_summary(res_dict, False)
    plot_performance_charts(res_dict, False)


def report() -> None:
    """Run a comprehensive report, saving to PDF.

    Run a comprehensive report for a specific timeframe, including portfolio P&L calculations, AUM retrieval,
    title page creation, summary generation, performance chart plotting, new trades analysis, exclusion of specific
    portfolios, best and worst performers analysis, pie chart plotting, combined pie chart plotting, metrics calculation,
    extraction of underlyings information, top holdings retrieval, ETF overlap analysis, and merging of PDF files.
    """
    logging.info(
        f"Running report for {args.timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    create_title_page(aum)
    get_summary(res_dict, True)
    plot_performance_charts(res_dict, True)
    res_dict.pop("Benchmark", None)
    create_new_trades_page(res_dict)
    create_best_and_worst_page(res_dict)
    create_best_and_worst_combined_page(res_dict)
    plot_pie_charts(res_dict)
    plot_combined_pie_chart(res_dict)
    create_metrics_page(res_dict)
    under_dict = get_underlyings(res_dict)
    if len(under_dict) > 1:
        create_top_holdings_page(res_dict, under_dict)
        create_overlaps_page(res_dict, under_dict)
    create_descriptions_page()
    file = config.get("Output", "file")
    pdf.merge_pdfs(saved_pdf_files, f"{output_dir}/{file}")
    logging.info("Complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments()
    start_date = args.start_date
    end_date = args.end_date
    output_dir = f"{args.path}/data/output"
    config = configparser.ConfigParser()
    config.read(f"{args.path}/{args.config}")
    report() if args.report else summary()
