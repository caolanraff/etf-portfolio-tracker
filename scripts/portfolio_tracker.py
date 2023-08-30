"""
ETF Portfolio Tracker.

This script tracks and analyses multiple ETF portfolios, given an Excel file with the trades made.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import argparse
import configparser
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfrw
import quantstats as qs
import requests
import seaborn as sns
import yfinance as yf
from bs4 import BeautifulSoup
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

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

files = []
output_dir = args.path + "/data/output/"
ticker_data = {}
mark_price = "Adj Close"


def get_ticker_data(ticker):
    """
    Retrieve historical data for a given ticker symbol.

    Args:
        ticker (str): Ticker symbol for the desired ETF.
    Returns:
        pd.DataFrame: DataFrame containing the historical data for the specified ticker.
    """
    if ticker in ticker_data.keys():
        return ticker_data[ticker]
    try:
        data = yf.download(ticker, progress=False)
    except Exception as e:
        logging.fatal(f"Unable to get data from Yahoo finance for {ticker}: {e}")
        sys.exit()
    data.index = pd.to_datetime(data.index).date
    data = data.reindex(pd.date_range(min(list(data.index)), end_date, freq="D"))
    data = data.fillna(method="ffill")
    ticker_data[ticker] = data
    return data


def calculate_portfolio_pnl(file_path, sheet):
    """
    Calculate the profit and loss (PnL) for a specific portfolio.

    Args:
        file_path (str): The path to the Excel file containing the portfolio data.
        sheet (str): The name of the sheet within the Excel file containing the portfolio data.
    Returns:
        pandas.DataFrame: A DataFrame containing the calculated PnL for the portfolio.
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


def calculate_all_portfolio_pnl():
    """
    Calculate the profit and loss (PnL) for all portfolios.

    Returns:
        dict: A dictionary containing the calculated PnL for each portfolio.
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


def save_dataframe_to_pdf(
    title,
    df,
    file,
    highlight_columns=None,
    thresholds=None,
    operators=None,
    highlight_colour=None,
):
    """
    Save a DataFrame as a PDF file with optional highlighting of cells based on specified conditions.

    Args:
        title (str): Title of the PDF document.
        df (pandas.DataFrame): The DataFrame to be saved as a PDF.
        file (str): The path and filename of the PDF file to be created.
        highlight_columns (list, optional): List of column names to be highlighted. Defaults to None.
        thresholds (list, optional): List of threshold values for highlighting. Defaults to None.
        operators (list, optional): List of comparison operators ('>' or '<') for highlighting. Defaults to None.
        highlight_colour (str, optional): The colour for highlighting the cells. Defaults to None.
    Returns:
        None
    """
    max_rows = 14
    if len(df) > max_rows:
        dfs = np.array_split(df, np.ceil(len(df) / max_rows))
        for i, sub_df in enumerate(dfs):
            new_file = f"{file[:-4]}_{i}.pdf"
            save_dataframe_to_pdf(
                title,
                sub_df,
                new_file,
                highlight_columns,
                thresholds,
                operators,
                highlight_colour,
            )
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", ha="left")
        else:
            cell.set_text_props(ha="left")
            if highlight_columns and thresholds and operators and highlight_colour:
                for i, col_name in enumerate(highlight_columns):
                    try:
                        col_index = df.columns.get_loc(col_name)
                    except KeyError:
                        raise ValueError(f"Column '{col_name}' not found in dataframe")
                    if col == col_index:
                        cell_value = float(cell.get_text().get_text())
                        if operators[i] == ">" and cell_value > thresholds[i]:
                            cell.set_facecolor(highlight_colour)
                        elif operators[i] == "<" and cell_value < thresholds[i]:
                            cell.set_facecolor(highlight_colour)

    ax.set_title(title, fontsize=12, fontweight="bold", y=0.9)
    pp = PdfPages(output_dir + file)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()
    files.append(output_dir + file)


def create_title_page(aum):
    """
    Create a title page for a PDF document with specified information.

    Args:
        aum (str): Assets Under Management (AUM) value to be displayed on the title page.
    Returns:
        None
    """
    pdf_output = FPDF()
    pdf_output.add_page()
    title = config.get("Text", "title")
    subtitle = end_date.strftime("%B %Y") + " Meeting"
    aum = "AUM: " + aum
    pdf_output.set_font("Arial", "B", 36)
    pdf_output.cell(0, 80, title, 0, 1, "C")
    pdf_output.set_font("Arial", "", 24)
    pdf_output.cell(0, 20, subtitle, 0, 1, "C")
    pdf_output.set_font("Arial", "", 16)
    pdf_output.cell(0, 20, aum, 0, 1, "C")
    image = config.get("Input", "image")
    if image != "":
        image_file = args.path + "/data/input/" + image
        pdf_output.image(image_file, x=55, y=150, w=100, h=100)
    files.append(output_dir + "title.pdf")
    pdf_output.output(files[-1])


def get_aum(result_dict):
    """
    Calculate the Assets Under Management (AUM) based on the portfolio values in the result dictionary.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
    Returns:
        str: The AUM value formatted as a string.
    """
    logging.info("Get AUM")
    portfolio_val = 0

    for name, df in result_dict.items():
        res = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].iloc[0]
        portfolio_val += res["portfolio_value"]

    aum = "$" + "{:,.0f}".format(portfolio_val)
    return aum


def get_summary(result_dict, save_to_file):
    """
    Calculate and displays or saves the summary information based on the result dictionary.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
        save_to_file (bool): Flag indicating whether to save the summary as a PDF file or print it.
    Returns:
        None
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
    summary.loc[0, "Notes"] = config.get("Text", "best")
    summary.loc[summary.index[-1], "Notes"] = config.get("Text", "worst")
    summary["Notes"] = summary["Notes"].fillna("")

    if save_to_file:
        save_dataframe_to_pdf("Summary", summary, "summary.pdf")
    else:
        print(summary)


def plot_performance_charts(result_dict, save_to_file):
    """
    Plot performance charts based on the result dictionary and optionally save them to a file.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
        save_to_file (bool): Flag indicating whether to save the performance charts as a PDF file or display them.
    Returns:
        None
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
        files.append(output_dir + "performance.pdf")
        plt.savefig(files[-1])
    else:
        plt.show()


def new_trades(result_dict):
    """
    Retrieve the new trades from the result dictionary and saves them as a PDF report.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
    Returns:
        None
    """
    logging.info("Getting new trades")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        trades = df.loc[df["quantity"] != 0]
        tickers = trades["ticker"].to_list()
        tickers = ", ".join(set(tickers))
        df = pd.DataFrame({"Portfolio": [key], "Trades": [tickers]})
        result_df = pd.concat([result_df, df], ignore_index=True)

    save_dataframe_to_pdf("New Trades", result_df, "new_trades.pdf")


def best_and_worst(result_dict):
    """
    Compute the best and worst performers among the ETFs in the result dictionary and saves the results as a PDF report.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
    Returns:
        None
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

        best_pnl_pct = merged_df.loc[merged_df["pnl_pct"].idxmax(), "ticker"]
        worst_pnl_pct = merged_df.loc[merged_df["pnl_pct"].idxmin(), "ticker"]
        best_price_pct = merged_df.loc[merged_df["price_pct"].idxmax(), "ticker"]
        worst_price_pct = merged_df.loc[merged_df["price_pct"].idxmin(), "ticker"]

        df = pd.DataFrame(
            {
                "Portfolio": [key],
                "Best PnL Pct": [best_pnl_pct],
                "Worst PnL Pct": [worst_pnl_pct],
                "Best Price Pct": [best_price_pct],
                "Worst Price Pct": [worst_price_pct],
            }
        )
        result_df = pd.concat([result_df, df], ignore_index=True)

    save_dataframe_to_pdf("Best & Worst Performers", result_df, "best_and_worst.pdf")


def best_and_worst_combined(result_dict):
    """
    Combine the best and worst performing ETFs based on their returns.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
    Returns:
        None
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

    save_dataframe_to_pdf(
        "Best & Worst Performers Combined", result_df, "best_and_worst_combined.pdf"
    )


def plot_pie_charts(result_dict):
    """
    Plot pie charts representing ETF weightings based on the result dictionary.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
    Returns:
        None
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
    files.append(output_dir + "weightings.pdf")
    plt.savefig(files[-1])


def plot_combined_pie_chart(result_dict):
    """
    Plot a combined pie chart representing the combined ETF weightings based on the result dictionary.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
    Returns:
        None
    """
    logging.info("Plotting combined ETF weightings")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].copy()
        result_df = pd.concat([result_df, df], ignore_index=True)

    df = result_df.groupby("ticker")["notional_value"].sum()
    total_sum = df.sum()
    other = float(config.get("Weightings", "other"))
    threshold = other * total_sum
    small_values = df[df < threshold]
    if len(small_values) > 0:
        df = df[df >= threshold]
        df["Other"] = small_values.sum()

    plt.clf()
    df.plot(kind="pie", autopct="%1.1f%%", colors=palette, textprops={"fontsize": 8})
    plt.title("Combined ETF Weightings", fontsize=12, fontweight="bold")
    plt.ylabel("")
    files.append(output_dir + "combined.pdf")
    plt.savefig(files[-1])


def get_yahoo_quote_table(ticker):
    """
    Scrapes data elements from Yahoo Finance's quote page for a given ticker.

    Args:
        ticker (str): Ticker symbol of the desired ETF.
    Returns:
        dict: Dictionary containing scraped data elements with attribute-value pairs.
    """
    url = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker
    try:
        tables = pd.read_html(
            requests.get(url, headers={"User-agent": "Mozilla/5.0"}).text
        )
    except Exception as e:
        logging.fatal(f"Unable to get metrics from Yahoo finance for {ticker}: {e}")
        sys.exit()
    data = pd.concat([tables[0], tables[1]])
    data.columns = ["attribute", "value"]
    data = data.sort_values("attribute")
    data = data.drop_duplicates().reset_index(drop=True)
    result = {key: val for key, val in zip(data.attribute, data.value)}
    return result


def calculate_sharpe_ratio(ticker):
    """
    Calculate the Sharpe ratio for a given ETF ticker.

    Args:
        ticker (str): The ETF ticker symbol.
    Returns:
        float: The Sharpe ratio.
    """
    data = get_ticker_data(ticker)
    min_date = end_date - timedelta(days=5 * 365)
    data = data.loc[min_date:end_date]
    returns = data[mark_price].pct_change()
    sharpe = qs.stats.sharpe(returns).round(2)
    return sharpe


def get_metrics(result_dict):
    """
    Retrieve and process metrics for the ETFs in the result dictionary.

    Args:
        result_dict (dict): A dictionary containing portfolio data as DataFrame objects.
    Returns:
        None
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
    threshold = config.get("Metrics", "threshold").split(",")
    operator = config.get("Metrics", "operator").split(",")
    highlight = config.get("Metrics", "highlight")

    if len(threshold) > 1:
        threshold = [float(s) for s in threshold]
        save_dataframe_to_pdf(
            "Metrics",
            result_df,
            "metrics.pdf",
            fields,
            threshold,
            operator,
            highlight,
        )
    else:
        save_dataframe_to_pdf("Metrics", result_df, "metrics.pdf")


def initcap(string):
    """
    Convert a string to initcap format.

    Args:
        string (str): The input string to be converted.
    Returns:
        str: The input string converted to initcap format, where the first letter of each word is capitalized.
    """
    words = string.lower().split()
    capitalized_words = [word.capitalize() for word in words]
    return " ".join(capitalized_words)


def extract_underlyings(tickers):
    """
    Extract underlying stock information for a list of tickers.

    Args:
        tickers (list): List of tickers for which to extract underlying stock information.
    Returns:
        pandas.DataFrame: DataFrame containing the extracted stock information, including ticker, stock symbol,
        company name, and weight.
    """
    df_list = []
    for ticker in tickers:
        url = f"https://www.zacks.com/funds/etf/{ticker}/holding"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"
        }
        with requests.Session() as req:
            req.headers.update(headers)
            r = req.get(url)
            html = r.text
            start = html.find("etf_holdings.formatted_data = ") + len(
                "etf_holdings.formatted_data = "
            )
            end = html.find(";", start)
            formatted_data = html[start:end].strip()
            try:
                data = json.loads(formatted_data)
            except Exception as e:
                logging.error(f"Unable to get underlyings for {ticker}: {e}")
                continue

            symbols = [
                item[1]
                if len(item[1]) <= 10
                else BeautifulSoup(item[1], "html.parser").find("a").get("rel")[0]
                if BeautifulSoup(item[1], "html.parser").find("a")
                else ""
                for item in data
            ]
            names = [
                re.search('title="([^"]+)"', item[0]).group(1).split("-", 1)[0]
                if "title=" in item[0]
                else item[0]
                for item in data
            ]
            weights = [float(lst[3]) if lst[3] != "NA" else None for lst in data]

            df = pd.DataFrame({"Stock": symbols, "Company": names, "Weight": weights})
            df.insert(0, "ticker", ticker)
            df["Company"] = df["Company"].apply(initcap)
            df_list.append(df)

    result_df = pd.concat(df_list, ignore_index=True)
    return result_df


def extract_all_underlyings(result_dict):
    """
    Extract underlying stock information for all tickers in the result_dict.

    Args:
        result_dict (dict): A dictionary containing the result data in the form of DataFrames, where the keys represent
        different categories and the values represent the corresponding DataFrames.
    Returns:
        dict: A dictionary containing the extracted stock information for each category, where the keys represent the
        categories and the values represent the corresponding DataFrames with ticker, stock symbol, company name, and
        weight information.
    """
    logging.info("Downloading ETF underlyings")
    underlyings_dict = {}
    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        tickers = df["ticker"].unique()
        underlyings = extract_underlyings(tickers)
        underlyings_dict[key] = underlyings
    return underlyings_dict


def get_top_holdings(result_dict, underlyings_dict):
    """
    Retrieve the top holdings based on the provided result_dict and underlyings_dict.

    Args:
        result_dict (dict): A dictionary containing the result data in the form of DataFrames, where the keys represent
        different categories and the values represent the corresponding DataFrames.
        underlyings_dict (dict): A dictionary containing the extracted stock information for each category, where the
        keys represent the categories and the values represent the corresponding DataFrames with ticker, stock symbol,
        company name, and weight information.
    Returns:
        pandas.DataFrame: DataFrame containing the top holdings information, including portfolio, number of stocks,
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
        res["symbol_notional"] = res["notional_value"] * (res["Weight"] / 100)
        grouped = res.groupby(["Stock", "Company"])["symbol_notional"].sum()
        grouped = grouped.reset_index()
        total_notional = df["notional_value"].sum()
        grouped["Weight"] = grouped["symbol_notional"] / total_notional * 100
        top = config.get("Holdings", "top")
        holdings = grouped.sort_values("Weight", ascending=False).head(int(top))
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

    save_dataframe_to_pdf("Top Holdings", result_df, "holdings.pdf")


def get_overlaps(result_dict, underlyings_dict):
    """
    Generate an ETF overlap heatmap based on the provided result_dict and underlyings_dict.

    Args:
        result_dict (dict): A dictionary containing the result data in the form of DataFrames, where the keys represent
        different categories and the values represent the corresponding DataFrames.
        underlyings_dict (dict): A dictionary containing the extracted stock information for each category, where the
        keys represent the categories and the values represent the corresponding DataFrames with ticker, stock symbol,
        company name, and weight information.
    Returns:
        None
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
        files.append(output_dir + "heatmap_" + key + ".pdf")
        pp = PdfPages(files[-1])
        pp.savefig(sns_plot.figure)
        pp.close()


def merge_pdfs(input_files, output_file):
    """
    Merge multiple PDF files into a single PDF file.

    Args:
        input_files (list): A list of input file paths (strings) representing the PDF files to be merged.
        output_file (str): The output file path (string) where the merged PDF file will be saved.
    Returns:
        None
    """
    logging.info("Merging files")
    pdf_output = pdfrw.PdfWriter()
    for file_name in input_files:
        pdf_input = pdfrw.PdfReader(file_name)
        for page in pdf_input.pages:
            pdf_output.addpage(page)
        os.remove(file_name)
    pdf_output.write(output_file)


def summary():
    """Run a summary report, printing the outputs.

    Run a report for a specific timeframe, calculates portfolio P&L, retrieves AUM (Assets Under Management),
    generates a summary, and plots performance charts.

    Returns:
        None
    """
    logging.info(
        f"Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    logging.info("AUM: " + aum)
    get_summary(res_dict, False)
    plot_performance_charts(res_dict, False)


def report():
    """Run a comprehensive report, saving to PDF.

    Run a comprehensive report for a specific timeframe, including portfolio P&L calculations, AUM retrieval,
    title page creation, summary generation, performance chart plotting, new trades analysis, exclusion of specific
    portfolios, best and worst performers analysis, pie chart plotting, combined pie chart plotting, metrics calculation,
    extraction of underlyings information, top holdings retrieval, ETF overlap analysis, and merging of PDF files.

    Returns:
        None
    """
    logging.info(
        f"Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    create_title_page(aum)
    get_summary(res_dict, True)
    plot_performance_charts(res_dict, True)
    new_trades(res_dict)
    best_and_worst(res_dict)
    best_and_worst_combined(res_dict)
    plot_pie_charts(res_dict)
    plot_combined_pie_chart(res_dict)
    get_metrics(res_dict)
    under_dict = extract_all_underlyings(res_dict)
    get_top_holdings(res_dict, under_dict)
    get_overlaps(res_dict, under_dict)
    merge_pdfs(files, output_dir + config.get("Output", "file"))


if __name__ == "__main__":
    if args.report:
        report()
    else:
        summary()
