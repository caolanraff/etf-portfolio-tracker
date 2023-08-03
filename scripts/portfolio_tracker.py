import argparse
import configparser
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfrw
import requests
import seaborn as sns
import yfinance as yf
from bs4 import BeautifulSoup
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages
from yahoo_fin import stock_info as si

parser = argparse.ArgumentParser()
parser.add_argument("--timeframe", type=str, help="timeframe [MTD|YTD|adhoc]")
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

files = []
output_dir = args.path + "/data/output/"


### Calculate pnl
def calculate_portfolio_pnl(file_path, sheet):
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
        ticker_data = yf.Ticker(ticker).history(
            start=min_date, end=max_date, auto_adjust=True
        )
        ticker_data = ticker_data.reindex(pd.date_range(min_date, max_date, freq="D"))
        ticker_data["Close"] = ticker_data["Close"].interpolate()
        prices[ticker] = ticker_data["Close"]

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


### Calculate PnL for all portfolios
def calculate_all_portfolio_pnl():
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


### Save dataframe to PDF
# @TODO: add title
def save_dataframe_to_pdf(
    title,
    df,
    file,
    highlight_columns=None,
    thresholds=None,
    operators=None,
    highlight_colour=None,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
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
    if title:
        ax.set_title(title)
    pp = PdfPages(file)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()


### Create title page
def create_title_page(aum):
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


### Get AUM
def get_aum(result_dict):
    logging.info("Get AUM")
    portfolio_val = 0

    for name, df in result_dict.items():
        res = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].iloc[0]
        portfolio_val += res["portfolio_value"]

    aum = "$" + "{:,.0f}".format(portfolio_val)
    return aum


### Get summary info
def get_summary(result_dict, save_to_file):
    logging.info("Get summary info")
    val = []

    for key, df in result_dict.items():
        som = df.loc[df["date"] == start_date].iloc[0]
        eom = df.loc[df["date"] == end_date].iloc[0]
        trades = df.loc[df["quantity"] != 0]
        pnl = (
            (eom["portfolio_pnl"] - som["portfolio_pnl"])
            / (som["portfolio_value"] + trades["total_cost"].sum())
        ) * 100
        val.append(pnl)

    summary = pd.DataFrame({"Investor": list(result_dict.keys()), timeframe: val})
    summary = summary.sort_values(by=timeframe, ascending=False)
    summary[timeframe] = summary[timeframe].round(3)
    summary = summary.reset_index(drop=True)
    summary.loc[0, "Notes"] = config.get("Text", "best")
    summary.loc[summary.index[-1], "Notes"] = config.get("Text", "worst")
    summary["Notes"] = summary["Notes"].fillna("")

    if save_to_file:
        files.append(output_dir + "summary.pdf")
        save_dataframe_to_pdf("Summary", summary, files[-1])
    else:
        print(summary)


### Chart portfolio performances
def plot_performance_charts(result_dict, save_to_file):
    logging.info("Plotting performance charts")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
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
        ax1.set_title("Overall PnL Change")
        ax1.set_xlim(start_date, end_date)

        (line2,) = ax2.plot(group["date"], group["pnl_pct_per_date"], label=name)
        if name not in labels:
            handles.append(line2)
            labels.append(name)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("PnL")
        ax2.set_title(timeframe + " PnL Change")
        ax2.set_xlim(start_date, end_date)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=14)
    fig.legend(handles, labels)

    if save_to_file:
        files.append(output_dir + "performance.pdf")
        plt.savefig(files[-1])
    else:
        plt.show()


### Get new trades
def new_trades(result_dict):
    logging.info("Getting new trades")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        trades = df.loc[df["quantity"] != 0]
        tickers = trades["ticker"].to_list()
        tickers = ", ".join(set(tickers))
        result_df = result_df.append(
            {"Investor": key, "Trades": tickers}, ignore_index=True
        )

    files.append(output_dir + "new_trades.pdf")
    save_dataframe_to_pdf("New Trades", result_df, files[-1])


### Get best and worst ETFs performance
def best_and_worst(result_dict):
    logging.info("Getting best and worst ETFs")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        start = df.loc[df["date"] == start_date]
        end = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        merged_df = pd.merge(start, end, on="ticker", suffixes=("_start", "_end"))
        total_notional = end["notional_value"].sum()

        merged_df["pnl_val"] = (
            (merged_df["total_pnl_end"] - merged_df["total_pnl_start"])
            / total_notional
            * 100
        )
        merged_df["pnl_pct"] = (
            (merged_df["total_pnl_end"] - merged_df["total_pnl_start"])
            / abs(merged_df["total_pnl_start"])
            * 100
        )

        best_portfolio_contribution = merged_df.loc[
            merged_df["pnl_val"].idxmax(), "ticker"
        ]
        worst_portfolio_contribution = merged_df.loc[
            merged_df["pnl_val"].idxmin(), "ticker"
        ]
        best_pnl_pct = merged_df.loc[merged_df["pnl_pct"].idxmax(), "ticker"]
        worst_pnl_pct = merged_df.loc[merged_df["pnl_pct"].idxmin(), "ticker"]

        result_df = result_df.append(
            {
                "Investor": key,
                "Best Portfolio Contributer": best_portfolio_contribution,
                "Worst Portfolio Contributer": worst_portfolio_contribution,
                "Best PnL %": best_pnl_pct,
                "Worst PnL %": worst_pnl_pct,
            },
            ignore_index=True,
        )

    files.append(output_dir + "best_and_worst.pdf")
    save_dataframe_to_pdf("Best & Worst Performers", result_df, files[-1])


### Plot ETF weightings pie chart
def plot_pie_charts(result_dict):
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
        axs[row][col].pie(
            df["notional_value"], labels=df["ticker"], autopct="%1.1f%%", radius=1.2
        )
        axs[row][col].set_title(
            key, y=1.1, fontdict={"fontsize": 14, "fontweight": "bold"}
        )

    for i in range(n, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row][col])

    plt.suptitle("ETF Weightings")
    files.append(output_dir + "weightings.pdf")
    plt.savefig(files[-1])


### Plot combined ETF weightings pie chart
def plot_combined_pie_chart(result_dict):
    logging.info("Plotting combined ETF weightings")
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].copy()
        result_df = result_df.append(df, ignore_index=True)

    df = result_df.groupby("ticker")["notional_value"].sum()
    total_sum = df.sum()
    other = float(config.get("Weightings", "other"))
    threshold = other * total_sum
    small_values = df[df < threshold]
    if len(small_values) > 0:
        df = df[df >= threshold]
        df["Other"] = small_values.sum()

    plt.clf()
    df.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Combined ETF Weightings", fontweight="bold")
    plt.ylabel("")
    files.append(output_dir + "combined.pdf")
    plt.savefig(files[-1])


### Get metrics
def get_metrics(result_dict):
    logging.info("Getting metrics")
    result_df = pd.DataFrame()
    metrics = [
        "Beta (5Y Monthly)",
        "Expense Ratio (net)",
        "PE Ratio (TTM)",
        "Yield",
        "YTD Daily Total Return",
    ]

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        tickers = list(df["ticker"].unique())
        for ticker in tickers:
            res = si.get_quote_table(ticker)
            my_dict = {k: v for k, v in res.items() if k in metrics}
            df = pd.DataFrame([my_dict])
            df["Investor"] = key
            df["Ticker"] = ticker
            result_df = result_df.append(df, ignore_index=True)

    result_df["Expense Ratio (net)"] = result_df["Expense Ratio (net)"].str.rstrip("%")
    result_df["Yield"] = result_df["Yield"].str.rstrip("%")
    result_df["YTD Daily Total Return"] = result_df[
        "YTD Daily Total Return"
    ].str.rstrip("%")
    result_df = result_df[["Investor", "Ticker"] + metrics]

    threshold = config.get("Metrics", "threshold").split(",")
    threshold = [float(s) for s in threshold]
    operator = config.get("Metrics", "operator").split(",")
    highlight = config.get("Metrics", "highlight")

    files.append(output_dir + "metrics.pdf")
    if len(threshold) > 1:
        save_dataframe_to_pdf(
            None, result_df, files[-1], metrics, threshold, operator, highlight
        )
    else:
        save_dataframe_to_pdf(None, result_df, files[-1])


### ETF holdings
def extract_underlyings(tickers):
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
            except:
                logging.error("Unable to get underlyings for " + ticker)
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
            df_list.append(df)

    result_df = pd.concat(df_list, ignore_index=True)
    return result_df


def extract_all_underlyings(result_dict):
    logging.info("Downloading ETF underlyings")
    underlyings_dict = {}
    for key, df in result_dict.items():
        tickers = df["ticker"].unique()
        underlyings = extract_underlyings(tickers)
        underlyings_dict[key] = underlyings
    return underlyings_dict


def get_top_holdings(result_dict, underlyings_dict):
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
        holdings["Investor"] = key
        stocks = underlyings["Stock"]
        holdings["No. of Stocks"] = len(stocks.unique())
        holdings["% of Overlap"] = round(
            100 * (len(stocks) - len(stocks.unique())) / len(stocks), 2
        )
        holdings["Weight"] = [round(x, 2) for x in holdings["Weight"]]
        holdings = holdings[
            ["Investor", "No. of Stocks", "% of Overlap", "Stock", "Company", "Weight"]
        ]
        result_df = result_df.append(holdings, ignore_index=True)

    files.append(output_dir + "holdings.pdf")
    save_dataframe_to_pdf(None, result_df, files[-1])


### ETF overlap heatmap
def get_overlaps(result_dict, underlyings_dict):
    logging.info("Plotting ETF overlap heatmap")

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
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
                overlaps = overlaps.append(
                    {"ETF1": k, "ETF2": i, "Overlap": val}, ignore_index=True
                )

        matrix = overlaps.pivot(index="ETF1", columns="ETF2", values="Overlap")
        matrix.index.name = None
        matrix.columns.name = None

        plt.clf()
        sns_plot = sns.heatmap(matrix, cmap="Blues", annot=True, fmt=".2f")
        sns_plot.figure.set_size_inches(10, 7)
        sns_plot.set_title("ETF Overlaps - " + key, fontsize=16)
        files.append(output_dir + "heatmap_" + key + ".pdf")
        pp = PdfPages(files[-1])
        pp.savefig(sns_plot.figure)
        pp.close()


### Merge pdf files
def merge_pdfs(input_files, output_file):
    logging.info("Merging files")
    pdf_output = pdfrw.PdfWriter()
    for file_name in input_files:
        pdf_input = pdfrw.PdfReader(file_name)
        for page in pdf_input.pages:
            pdf_output.addpage(page)
        os.remove(file_name)
    pdf_output.write(output_file)


### Main
def comp():
    logging.info(
        f"Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    logging.info("AUM: " + aum)
    get_summary(res_dict, False)
    plot_performance_charts(res_dict, False)


def report():
    logging.info(
        f"Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})"
    )
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    create_title_page(aum)
    get_summary(res_dict, True)
    plot_performance_charts(res_dict, True)
    new_trades(res_dict)
    exclude = config.get("Input", "exclude").split(",")
    for key in exclude:
        if key in res_dict:
            del res_dict[key]
    best_and_worst(res_dict)
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
        comp()
