"""
Functions required to generate report.

Author: Caolan Rafferty
Date: 2024-09-06
"""
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

from src.cli.const import CHART_PALETTE, MARK_PRICE
from src.utils.data import get_metrics, get_sector_weightings, get_ticker_info
from src.utils.pdf import df_to_pdf, save_paragraphs_to_pdf
from src.utils.types import DictFrame, Frame, Time

plt.style.use("seaborn-v0_8")


def create_title_page(
    title: str, aum: str, image_file: str, end_date: Time, output_dir: str
) -> str:
    """
    Create a title page for a PDF document with specified information.

    Parameters:
    title (str): The title of the document.
    aum (str): The Assets Under Management (AUM) value.
    image_file (str): The file path of an image to include on the title page.
    end_date (Time): The end date for the document, used to generate the subtitle.
    output_dir (str): The directory where the PDF file will be saved.

    Returns:
    str: The file path of the created title page PDF.
    """
    pdf_output = FPDF()
    pdf_output.add_page()
    subtitle = f"{end_date.strftime('%B %Y')} Meeting"
    aum = f"AUM: {aum}"
    pdf_output.set_font("Arial", "B", 36)
    pdf_output.cell(0, 80, title, 0, 1, "C")
    pdf_output.set_font("Arial", "", 24)
    pdf_output.cell(0, 20, subtitle, 0, 1, "C")
    pdf_output.set_font("Arial", "", 16)
    pdf_output.cell(0, 20, aum, 0, 1, "C")
    if image_file != "":
        pdf_output.image(image_file, x=55, y=150, w=100, h=100)
    file = f"{output_dir}/title.pdf"
    pdf_output.output(file)
    return file


def create_new_trades_page(result_dict: DictFrame, output_dir: str) -> list[str]:
    """
    Retrieve the new trades from the result dictionary and save them as a PDF report.

    Parameters:
    result_dict (DictFrame): A dictionary containing the trade results, where each key is a portfolio name and each value is a DataFrame of trades.
    output_dir (str): The directory where the PDF report will be saved.

    Returns:
    list[str]: The file paths of the created PDFs.
    """
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

    files = df_to_pdf("New Trades", result_df, output_dir)
    return files


def create_best_and_worst_page(
    result_dict: DictFrame, end_date: Time, output_dir: str
) -> list[str]:
    """
    Compute the best and worst performers among the ETFs in the result dictionary and saves the results as a PDF report.

    Parameters:
    result_dict (DictFrame): A dictionary containing portfolio data as DataFrame objects.
    end_date (Time): The end date for calculating the performance.
    output_dir (str): The directory where the PDF report will be saved.

    Returns:
    list[str]: The file paths of the created PDFs.
    """
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

    files = df_to_pdf("Best & Worst Performers", result_df, output_dir)
    return files


def create_best_and_worst_combined_page(
    result_dict: DictFrame,
    ticker_data: DictFrame,
    start_date: Time,
    end_date: Time,
    output_dir: str,
) -> list[str]:
    """
    Combine the best and worst performing ETFs based on their returns.

    Parameters:
    result_dict: A dictionary containing ETF data.
    ticker_data: A dictionary containing ticker data.
    start_date: The start date for analysis.
    end_date: The end date for analysis.
    output_dir: The directory to save the output PDF file.

    Returns:
    list[str]: The file paths of the created PDFs.
    """
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
        df = df.loc[
            (df["date"].between(start_date, end_date)) & (df["cumulative_quantity"] > 0)
        ]
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

    files = df_to_pdf("Best & Worst Performers Combined", result_df, output_dir)
    return files


def create_descriptions_page(tickers: list[str], output_dir: str) -> str:
    """
    Create ETF descriptions page.

    Parameters:
    tickers (list[str]): A list of ticker symbols for which to create descriptions.
    output_dir (str): The directory where the descriptions PDF will be saved.

    Returns:
    str: The file path of the created PDF.
    """
    headers = []
    paragraphs = []

    for i in tickers:
        data = get_ticker_info(i)
        name = data["name"]
        headers += [f"{name} ({i})"]
        paragraphs += [data["description"]]

    file = save_paragraphs_to_pdf("ETF Descriptions", headers, paragraphs, output_dir)
    return file


def create_overlaps_page(
    result_dict: DictFrame, underlyings_df: Frame, output_dir: str
) -> list[str]:
    """
    Generate an ETF overlap heatmap based on the provided result_dict.

    Parameters:
    result_dict (DictFrame): A dictionary containing result data as DataFrames, where keys represent different categories.
    output_dir (str): Directory path to save the generated heatmap PDF files.

    Returns:
    list[str]: The file paths of the created PDFs.
    """
    file_list = []
    for key, df in result_dict.items():
        underlyings = underlyings_df[
            underlyings_df["ticker"].isin(df["ticker"].unique())
        ].copy()
        if underlyings.empty:
            continue
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
        file_list.append(file)

    return file_list


def get_aum(result_dict: DictFrame, end_date: Time) -> str:
    """
    Calculate the Assets Under Management (AUM) based on the portfolio values in the result dictionary.

    Parameters:
    result_dict (DictFrame): A dictionary containing portfolio data as DataFrame objects.
    end_date (Time): The end date of the report.

    Returns:
    str: The AUM value.
    """
    portfolio_val = []

    for name, df in result_dict.items():
        if name == "Benchmark":
            continue
        res = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].iloc[0]
        portfolio_val += [res["portfolio_value"]]

    aum = f"${sum(portfolio_val):,.0f}"
    return aum


def plot_performance_charts(
    args: Any, result_dict: DictFrame, output_dir: str = ""
) -> Any:
    """
    Plot performance charts based on the result dictionary and optionally save them to a file.

    Parameters:
    args (Any): Arguments containing the start date, end date, and timeframe for the plots.
    result_dict (DictFrame): A dictionary containing portfolio data as DataFrame objects.
    output_dir (str): Directory path to save the performance charts as a PDF file. If not provided, the charts will be displayed instead.

    Returns:
    Any: The file path of the saved PDF if `output_dir` is provided, otherwise None.
    """
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

        ax2.plot(group["date"], group["pnl_pct_per_date"], label=name)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("PnL")
        ax2.set_title(f"{args.timeframe} PnL Change", fontsize=12, fontweight="bold")
        ax2.set_xlim(args.start_date, args.end_date)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=14)
    fig.legend(handles, labels)

    if output_dir != "":
        file = f"{output_dir}/performance.pdf"
        plt.savefig(file)
        return file
    else:
        plt.show()


def plot_combined_pie_chart(
    result_dict: DictFrame, end_date: Time, other_threshold: float, output_dir: str
) -> str:
    """
    Plot a combined pie chart representing the combined ETF weightings based on the result dictionary.

    Args:
    result_dict (DictFrame): A dictionary containing portfolio data as DataFrame objects.
    end_date (Time): The date for which the ETF weightings are calculated.
    other_threshold (float): The threshold percentage for grouping small values under 'Other'.
    output_dir (str): The directory to save the generated pie chart PDF.

    Returns:
    str: The file path of the saved pie chart PDF.
    """
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)].copy()
        result_df = pd.concat([result_df, df], ignore_index=True)

    df = result_df.groupby("ticker")["notional_value"].sum()
    total_sum = df.sum()
    threshold = other_threshold * total_sum
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
    return file


def plot_pie_charts(
    result_dict: DictFrame, end_date: Time, other_threshold: float, output_dir: str
) -> str:
    """
    Plot pie charts representing ETF weightings based on the result dictionary.

    Args:
    result_dict (DictFrame): A dictionary containing portfolio data as DataFrame objects.
    end_date (Time): The date or timestamp for which the ETF weightings are calculated.
    other_threshold (float): The threshold percentage for grouping small values under 'Other'.
    output_dir (str): The directory path where the pie charts will be saved as a PDF file.

    Returns:
    str: The file path of the saved PDF containing the pie charts.
    """
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
        threshold = other_threshold * total_sum
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
    return file


def create_metrics_page(
    result_dict: DictFrame,
    end_date: Time,
    threshold: list[str],
    operator: list[str],
    highlight: str,
    output_dir: str,
) -> list[str]:
    """
    Retrieve and process metrics for the ETFs in the result dictionary, and save the results as a PDF.

    Parameters:
    result_dict (DictFrame): A dictionary containing portfolio data as DataFrame objects.
    end_date (Time): The end date for calculating metrics.
    threshold (list[str]): List of threshold values for highlighting in the PDF.
    operator (list[str]): List of comparison operators ('>' or '<') for highlighting in the PDF.
    highlight (str): The color for highlighting the cells in the PDF.
    output_dir (str): The directory where the output PDF will be saved.

    Returns:
    list[str]: The file paths of the created PDFs.
    """
    tickers = list(set().union(*[df["ticker"] for df in result_dict.values()]))
    metrics = get_metrics(tickers)

    df_list = []
    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        tickers = list(df["ticker"].unique())
        data = metrics[metrics["Ticker"].isin(tickers)].copy()
        data.insert(0, "Portfolio", key)
        df_list.append(data)

    result_df = pd.concat(df_list, ignore_index=True)
    result_df = result_df.fillna("-")
    fields = [s for s in result_df.columns if s not in ["Portfolio", "Ticker"]]

    if threshold[0] != "":
        files = df_to_pdf(
            "Metrics",
            result_df,
            output_dir,
            fields,
            [float(s) for s in threshold],
            operator,
            highlight,
        )
    else:
        files = df_to_pdf("Metrics", result_df, output_dir)

    return files


def get_summary(
    result_dict: DictFrame,
    start_date: Time,
    end_date: Time,
    timeframe: str,
    comments: dict[str, str],
    output_dir: str = "",
) -> Any:
    """
    Retrieve and process metrics for the ETFs in the result dictionary, and save the results as a PDF.

    Parameters:
    result_dict (DictFrame): A dictionary containing portfolio data as DataFrame objects.
    end_date (Time): The end date for calculating metrics.
    threshold (list[str]): List of threshold values for highlighting in the PDF.
    operator (list[str]): List of comparison operators ('>' or '<') for highlighting in the PDF.
    highlight (str): The color for highlighting the cells in the PDF.
    output_dir (str): The directory where the output PDF will be saved.

    Returns:
    Any: The file paths of the PDFs if `output_dir` is provided, otherwise None.
    """
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

    summary = pd.DataFrame({"Portfolio": list(result_dict.keys()), timeframe: val})
    summary = summary.sort_values(by=timeframe, ascending=False)
    summary[timeframe] = summary[timeframe].round(3)
    summary = summary.reset_index(drop=True)
    summary["Notes"] = ""

    if comments is not None:
        summary.loc[0, "Notes"] = comments.get("best", "")
        summary.loc[summary.index[-1], "Notes"] = comments.get("worst", "")

    if output_dir != "":
        files = df_to_pdf("Summary", summary, output_dir)
        return files
    else:
        print(summary)


def create_top_holdings_page(
    result_dict: DictFrame,
    underlyings_df: Frame,
    end_date: Time,
    num_of_companies: int,
    threshold: float,
    output_dir: str,
) -> list[str]:
    """
    Generate a PDF report of the top holdings based on the provided result_dict and underlyings data.

    Parameters:
    result_dict (DictFrame): A dictionary containing the result data in the form of DataFrames, where the keys represent
        different categories and the values represent the corresponding DataFrames.
    end_date (Time): The end date for filtering the data.
    num_of_companies (int): The number of top companies to include in the report.
    threshold (float): The threshold value for highlighting weights in the PDF.
    output_dir (str): The directory where the output PDF file will be saved.

    Returns:
    list[str]: The file paths of the created PDFs.
    """
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        df = df[["ticker", "notional_value"]]
        underlyings = underlyings_df[
            underlyings_df["ticker"].isin(df["ticker"].unique())
        ].copy()
        underlyings = underlyings.drop_duplicates(subset=["ticker", "Stock", "Company"])
        if underlyings.empty:
            continue
        res = pd.merge(df, underlyings, on=["ticker"], how="left")
        res["Company"] = res["Company"].str.rstrip(".")
        res["symbol_notional"] = res["notional_value"] * (res["Weight"] / 100)
        grouped = res.groupby(["Stock", "Company"])["symbol_notional"].sum()
        grouped = grouped.reset_index()
        total_notional = df["notional_value"].sum()
        grouped["Weight"] = grouped["symbol_notional"] / total_notional * 100
        holdings = grouped.sort_values("Weight", ascending=False).head(num_of_companies)
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

    if threshold > 0.0:
        files = df_to_pdf(
            "Top Holdings",
            result_df,
            output_dir,
            ["Weight"],
            [threshold],
            [">"],
            "red",
        )
    else:
        files = df_to_pdf("Top Holdings", result_df, output_dir)

    return files


def plot_sector_weightings_page(
    result_dict: DictFrame, end_date: Time, output_dir: str
) -> str:
    """
    Generate saves a multi-panel pie chart visualization of sector weightings for a given set of portfolios.

    Parameters:
    result_dict (DictFrame): A dictionary where keys are portfolio names and values are DataFrames containing portfolio
        holdings.
    end_date (Time): The date for which sector weightings are calculated.
    output_dir (str): The directory where the generated PDF file will be saved.

    Returns:
    str: The file path of the saved sector weightings plot (PDF format).
    """
    tickers = list(set().union(*[df["ticker"] for df in result_dict.values()]))
    sectors = get_sector_weightings(tickers)

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
        df = df.loc[(df["date"] == end_date) & (df["cumulative_quantity"] > 0)]
        df = df[["ticker", "notional_value"]].rename(columns={"ticker": "Ticker"})
        tickers = list(df["Ticker"].unique())
        data = sectors[sectors["Ticker"].isin(tickers)].copy()

        res = pd.merge(data, df, on="Ticker", how="left")
        res["sector_value"] = res["Weight"] * res["notional_value"]
        res = res.groupby("Sector")["sector_value"].sum().reset_index()
        res["sector_weight"] = res["sector_value"] / res["sector_value"].sum()

        row = i // num_cols
        col = i % num_cols
        axs[row][col].set_prop_cycle(color=CHART_PALETTE)
        axs[row][col].pie(
            res["sector_weight"].to_list(),
            labels=res["Sector"].to_list(),
            autopct="%1.1f%%",
            radius=1.2,
            textprops={"fontsize": 6},
        )
        axs[row][col].set_title(
            key, y=1.1, fontdict={"fontsize": 10, "fontweight": "bold"}
        )

    for i in range(n, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row][col])

    plt.suptitle("Sector Weightings", fontsize=12, fontweight="bold")
    file = f"{output_dir}/sectors.pdf"
    plt.savefig(file)
    return file
