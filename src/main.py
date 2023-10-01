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
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs
import seaborn as sns
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

import utils.util as util
from utils.data import (
    get_etf_underlyings,
    get_ticker_data,
    get_yahoo_quote_table,
    ticker_data,
)
from utils.util import saved_pdf_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--timeframe", default="MTD", type=str, help="timeframe [MTD|YTD|adhoc]"
)
parser.add_argument("--start", default="", type=str, help="start date")
parser.add_argument("--end", default="", type=str, help="end date")
parser.add_argument("--report", action="store_true", help="generate PDF report")
parser.add_argument("--path", default="./", type=str, help="directory path")
parser.add_argument(
    "--config", default="config/default.ini", type=str, help="config file"
)
args = parser.parse_args()

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

DictFrame = Dict[str, pd.DataFrame]

timeframe = args.timeframe
start_date = args.start
end_date = args.end
now = datetime.now()
if start_date == "":
    if timeframe == "MTD":
        start_date = datetime(now.year, now.month, 1)
    elif timeframe == "YTD":
        start_date = datetime(now.year, 1, 1)
    else:
        logging.error("Unknown timeframe")
        exit()
else:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

start_date = start_date + timedelta(days=-1)

if end_date == "":
    end_date = datetime(now.year, now.month, now.day)
else:
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

config = configparser.ConfigParser()
config.read(args.path + "/" + args.config)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
plt.style.use("seaborn")
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

output_dir = args.path + "/data/output/"
mark_price = "Adj Close"


def calculate_portfolio_pnl(file_path: str, sheet: str) -> pd.DataFrame:
    """
    Calculate the profit and loss (PnL) for a specific portfolio.

    Args:
        file_path: The path to the Excel file containing the portfolio data.
        sheet: The name of the sheet within the Excel file containing the portfolio data.
    Returns:
        A DataFrame containing the calculated PnL for the portfolio.
    """
    df = pd.read_excel(file_path, sheet_name=sheet)
    if len(df) == 0:
        logging.warning("Tab is empty for " + sheet)
        return

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
        # Calculate running cumulative quantity and running average entry price
        merged_df["cumulative_quantity"] = merged_df["quantity"].cumsum()
        merged_df["total_cost"] = merged_df["quantity"] * merged_df["price"]
        merged_df["cumulative_cost"] = merged_df["total_cost"].cumsum()
        # Handle sells
        merged_df["average_entry_price"] = np.where(
            merged_df["cumulative_quantity"] == 0,
            np.nan,
            merged_df["cumulative_cost"] / merged_df["cumulative_quantity"],
        )
        merged_df = merged_df.groupby("ticker").apply(
            lambda x: x.fillna(method="ffill")
        )
        result_df = pd.concat([result_df, merged_df], ignore_index=True)

    # Download the adjusted close price from Yahoo Finance for each ticker and date
    min_date = result_df["date"].min()
    min_date = min_date - timedelta(days=7)
    prices = {}
    for ticker in result_df["ticker"].unique():
        data = get_ticker_data(ticker)
        data = data.loc[min_date:max_date][mark_price]
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
    # Calculate the portfolio value per date
    result_df["portfolio_cost"] = result_df.groupby("date")[
        "cumulative_cost"
    ].transform("sum")
    result_df["portfolio_value"] = result_df.groupby("date")[
        "notional_value"
    ].transform("sum")
    # Calculate the PNL percentage
    result_df["pnl_pct"] = (
        100
        * (result_df["portfolio_value"] - result_df["portfolio_cost"])
        / result_df["portfolio_cost"]
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
    file = args.path + "/data/input/" + config.get("Input", "file")
    sheets = pd.ExcelFile(file).sheet_names
    for sheet in sheets:
        res = calculate_portfolio_pnl(file, sheet)
        if res is None:
            continue
        res = res[(res["date"] >= start_date) & (res["date"] <= end_date)]
        group = res.groupby("ticker")["cumulative_quantity"].sum()
        tickers = group[group == 0].index
        res = res[~res["ticker"].isin(tickers)]
        result_dict[sheet] = res
    return result_dict


def create_title_page(aum: str) -> None:
    """
    Create a title page for a PDF document with specified information.

    Args:
        aum: Assets Under Management (AUM) value to be displayed on the title page.
    """
    pdf_output = FPDF()
    pdf_output.add_page()
    title = config.get("TitlePage", "title")
    subtitle = end_date.strftime("%B %Y") + " Meeting"
    aum = "AUM: " + aum
    pdf_output.set_font("Arial", "B", 36)
    pdf_output.cell(0, 80, title, 0, 1, "C")
    pdf_output.set_font("Arial", "", 24)
    pdf_output.cell(0, 20, subtitle, 0, 1, "C")
    pdf_output.set_font("Arial", "", 16)
    pdf_output.cell(0, 20, aum, 0, 1, "C")
    image = config.get("TitlePage", "image")
    if image != "":
        image_file = args.path + "/data/input/" + image
        pdf_output.image(image_file, x=55, y=150, w=100, h=100)
    file = output_dir + "title.pdf"
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
    logging.info("Get AUM")
    portfolio_val = 0

    for name, df in result_dict.items():
        res = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].iloc[0]
        portfolio_val += res["portfolio_value"]

    aum = "$" + "{:,.0f}".format(portfolio_val)
    return aum


def get_summary(result_dict: DictFrame, save_to_file: bool) -> None:
    """
    Calculate and displays or saves the summary information based on the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
        save_to_file: Flag indicating whether to save the summary as a PDF file or print it.
    """
    logging.info("Get summary info")
    val = []

    for key, df in result_dict.items():
        start = max(min(df["date"]).to_pydatetime(), start_date)
        first_day = df.loc[df["date"] == start].iloc[0]
        last_day = df.loc[df["date"] == end_date].iloc[0]
        trades = df.loc[df["quantity"] != 0]
        pnl = (
            (last_day["portfolio_pnl"] - first_day["portfolio_pnl"])
            / (first_day["portfolio_value"] + trades["total_cost"].sum())
        ) * 100
        val.append(pnl)

    summary = pd.DataFrame({"Portfolio": list(result_dict.keys()), timeframe: val})
    summary = summary.sort_values(by=timeframe, ascending=False)
    summary[timeframe] = summary[timeframe].round(3)
    summary = summary.reset_index(drop=True)
    summary.loc[0, "Notes"] = config.get("SummaryPage", "best")
    summary.loc[summary.index[-1], "Notes"] = config.get("SummaryPage", "worst")
    summary["Notes"] = summary["Notes"].fillna("")

    if save_to_file:
        util.df_to_pdf("Summary", summary, output_dir + "summary.pdf")
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
        group = df.groupby("date").agg(
            {
                "pnl_pct": "first",
                "portfolio_value": "first",
                "portfolio_pnl": "first",
                "total_cost": "sum",
            }
        )
        group["pnl_pct_per_date"] = (
            (group["portfolio_pnl"] - group["portfolio_pnl"].iloc[0])
            / (group["portfolio_value"].iloc[0] + group["total_cost"].cumsum())
            * 100
        )
        group = group.reset_index()

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
        ax2.set_title(timeframe + " PnL Change", fontsize=12, fontweight="bold")
        ax2.set_xlim(start_date, end_date)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=14)
    fig.legend(handles, labels)

    if save_to_file:
        file = output_dir + "performance.pdf"
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

    util.df_to_pdf("New Trades", result_df, output_dir + "new_trades.pdf")


def create_best_and_worst_page(result_dict: DictFrame) -> None:
    """
    Compute the best and worst performers among the ETFs in the result dictionary and saves the results as a PDF report.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Getting best and worst ETFs")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        start = max(min(df["date"]).to_pydatetime(), start_date)
        first_day = df.loc[df["date"] == start]
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

        df = pd.DataFrame(
            {
                "Portfolio": [key],
                "Best Price Pct": [best_price_pct],
                "Worst Price Pct": [worst_price_pct],
                "Best PnL Pct": [best_pnl_pct],
                "Worst PnL Pct": [worst_pnl_pct],
            }
        )
        result_df = pd.concat([result_df, df], ignore_index=True)

    util.df_to_pdf(
        "Best & Worst Performers", result_df, output_dir + "best_and_worst.pdf"
    )


def create_best_and_worst_combined_page(result_dict: DictFrame) -> None:
    """
    Combine the best and worst performing ETFs based on their returns.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    returns = pd.DataFrame(columns=["Ticker", "Returns"])
    result_df = pd.DataFrame()
    tickers = []

    for key, df in ticker_data.items():
        df = df.loc[start_date:end_date]
        first_close = df[mark_price].iloc[0]
        last_close = df[mark_price].iloc[-1]
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
        lambda row: str(row["Ticker"]) + " (" + str(row["Returns"]) + "%)", axis=1
    )
    bottom = returns.sort_values(by="Returns", ascending=True)
    bottom = bottom.head(5)
    bottom.reset_index(drop=True, inplace=True)
    result_df["Bottom 5 ETFs"] = bottom.apply(
        lambda row: str(row["Ticker"]) + " (" + str(row["Returns"]) + "%)", axis=1
    )

    util.df_to_pdf(
        "Best & Worst Performers Combined",
        result_df,
        output_dir + "best_and_worst_combined.pdf",
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
        axs[row][col].set_prop_cycle(color=palette)
        axs[row][col].pie(
            df["notional_value"],
            labels=df["ticker"],
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
    file = output_dir + "weightings.pdf"
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
    if len(small_values) > 0:
        df = df[df >= threshold]
        df["Other"] = small_values.sum()

    plt.clf()
    df.plot(kind="pie", autopct="%1.1f%%", colors=palette, textprops={"fontsize": 8})
    plt.title("Combined ETF Weightings", fontsize=12, fontweight="bold")
    plt.ylabel("")
    file = output_dir + "combined.pdf"
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
    returns = data[mark_price].pct_change()
    sharpe = qs.stats.sharpe(returns).round(2)
    return float(sharpe)


def create_metrics_page(result_dict: DictFrame) -> None:
    """
    Retrieve and process metrics for the ETFs in the result dictionary.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
    """
    logging.info("Getting metrics")
    result_df = pd.DataFrame()
    metrics_mapping = {
        "Beta (5Y Monthly)": "Beta",
        "Expense Ratio (net)": "Expense Ratio",
        "PE Ratio (TTM)": "PE Ratio",
        "Yield": "Dividend Yield",
        "YTD Daily Total Return": "YTD",
    }

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        tickers = list(df["ticker"].unique())
        for ticker in tickers:
            res = get_yahoo_quote_table(ticker)
            my_dict = {
                k: v for k, v in res.items() if k in list(metrics_mapping.keys())
            }
            df = pd.DataFrame([my_dict])
            df["Portfolio"] = key
            df["Ticker"] = ticker
            df["Sharpe Ratio"] = calculate_sharpe_ratio(ticker)
            result_df = pd.concat([result_df, df], ignore_index=True)

    result_df["Expense Ratio (net)"] = result_df["Expense Ratio (net)"].str.rstrip("%")
    result_df["Yield"] = result_df["Yield"].str.rstrip("%")
    result_df["YTD Daily Total Return"] = result_df[
        "YTD Daily Total Return"
    ].str.rstrip("%")
    result_df = result_df[
        ["Portfolio", "Ticker", "Sharpe Ratio"] + list(metrics_mapping.keys())
    ]
    result_df = result_df.rename(columns=metrics_mapping)

    fields = ["Sharpe Ratio"] + list(metrics_mapping.values())
    threshold = config.get("MetricsPage", "threshold").split(",")
    operator = config.get("MetricsPage", "operator").split(",")
    highlight = config.get("MetricsPage", "highlight")

    file = output_dir + "metrics.pdf"
    if len(threshold) > 1:
        util.df_to_pdf(
            "Metrics",
            result_df,
            file,
            fields,
            [float(s) for s in threshold],
            operator,
            highlight,
        )
    else:
        util.df_to_pdf("Metrics", result_df, file)


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

    file = output_dir + "holdings.pdf"
    if len(threshold) > 0:
        util.df_to_pdf(
            "Top Holdings",
            result_df,
            file,
            ["Weight"],
            [float(threshold)],
            [">"],
            "red",
        )
    else:
        util.df_to_pdf("Top Holdings", result_df, file)


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
        sns_plot.set_title("ETF Overlaps - " + key, fontsize=12, fontweight="bold")
        file = output_dir + "heatmap_" + key + ".pdf"
        pp = PdfPages(file)
        pp.savefig(sns_plot.figure)
        pp.close()
        saved_pdf_files.append(file)


def summary() -> None:
    """Run a summary report, printing the outputs.

    Run a report for a specific timeframe, calculates portfolio P&L, retrieves AUM (Assets Under Management),
    generates a summary, and plots performance charts.
    """
    logging.info(
        f"Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    logging.info("AUM: " + aum)
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
        f"Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    create_title_page(aum)
    get_summary(res_dict, True)
    plot_performance_charts(res_dict, True)
    create_new_trades_page(res_dict)
    create_best_and_worst_page(res_dict)
    create_best_and_worst_combined_page(res_dict)
    plot_pie_charts(res_dict)
    plot_combined_pie_chart(res_dict)
    create_metrics_page(res_dict)
    under_dict = get_underlyings(res_dict)
    create_top_holdings_page(res_dict, under_dict)
    create_overlaps_page(res_dict, under_dict)
    util.merge_pdfs(saved_pdf_files, output_dir + config.get("Output", "file"))
    logging.info("Complete")


if __name__ == "__main__":
    if args.report:
        report()
    else:
        summary()
