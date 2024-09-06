"""
Functions required to generate report.

Author: Caolan Rafferty
Date: 2024-09-06
"""

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

from src.cli.const import MARK_PRICE
from src.utils.data import get_ticker_info
from src.utils.pdf import df_to_pdf, save_paragraphs_to_pdf
from src.utils.types import DictFrame, Time


def create_title_page(
    title: str, aum: str, image_file: str, end_date: Time, output_dir: str
) -> str:
    """
    Create a title page for a PDF document with specified information.

    Args:
        aum: Assets Under Management (AUM) value to be displayed on the title page.
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


def create_new_trades_page(
    result_dict: Dict[str, pd.DataFrame], output_dir: str
) -> None:
    """
    Retrieve the new trades from the result dictionary and saves them as a PDF report.

    Args:
        result_dict: A dictionary containing portfolio data as DataFrame objects.
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

    df_to_pdf("New Trades", result_df, f"{output_dir}/new_trades.pdf")


def create_best_and_worst_page(
    result_dict: DictFrame, end_date: Time, output_dir: str
) -> None:
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

    df_to_pdf("Best & Worst Performers", result_df, f"{output_dir}/best_and_worst.pdf")


def create_best_and_worst_combined_page(
    result_dict: DictFrame,
    ticker_data: DictFrame,
    start_date: Time,
    end_date: Time,
    output_dir: str,
) -> None:
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

    df_to_pdf(
        "Best & Worst Performers Combined",
        result_df,
        f"{output_dir}/best_and_worst_combined.pdf",
    )


def create_descriptions_page(tickers: list[str], output_dir: str) -> None:
    """Create ETF descriptions page."""
    logging.info("Creating description page")

    headers = []
    paragraphs = []

    for i in tickers:
        data = get_ticker_info(i)
        if "longBusinessSummary" in data:
            name = data["shortName"]
            headers += [f"{name} ({i})"]
            paragraphs += [data["longBusinessSummary"]]

    save_paragraphs_to_pdf(
        "ETF Descriptions", headers, paragraphs, f"{output_dir}/descriptions.pdf"
    )


def create_overlaps_page(
    result_dict: DictFrame, underlyings_dict: DictFrame, output_dir: str
) -> list[str]:
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

    file_list = []
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
        file_list.append(file)

    return file_list
