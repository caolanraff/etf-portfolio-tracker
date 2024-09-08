"""
Calculations required for report generation.

Author: Caolan Rafferty
Date: 2024-09-06
"""
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import quantstats as qs

from src.cli.const import MARK_PRICE
from src.utils.data import get_ticker_data
from src.utils.types import DictFrame, Frame, Time


def calculate_entry_price(df: Frame) -> Frame:
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


def calculate_costs_and_proceeds(ticker: str, df: Frame, end_date: Time) -> Frame:
    """
    Calculate costs and proceeds for a given ticker.

    This function calculates the cumulative quantity, total cost, cumulative cost, total proceeds,
    and cumulative proceeds based on the provided DataFrame.
    It also incorporates the calculation of the average entry price using the 'calculate_entry_price' function.
    """
    date_range = pd.date_range(
        start=df["date"].min(), end=pd.Timestamp(end_date), freq="D"
    )
    res = pd.DataFrame({"date": date_range, "ticker": ticker})
    res = pd.merge(res, df, on=["date", "ticker"], how="outer")
    res = res.fillna({"quantity": 0, "price": 0})

    res["cumulative_quantity"] = res["quantity"].cumsum()
    res["total_cost"] = np.where(res["quantity"] > 0, res["quantity"] * res["price"], 0)
    res["cumulative_cost"] = res["total_cost"].cumsum()
    res["total_proceeds"] = np.where(
        res["quantity"] < 0, res["quantity"] * res["price"], 0
    )
    res["cumulative_proceeds"] = res["total_proceeds"].cumsum()

    entry_price = calculate_entry_price(res)
    res = pd.merge(res, entry_price, on="date", how="left")
    res = res.groupby("ticker").apply(lambda x: x.ffill())

    return res


def calculate_portfolio_pnl(df: Frame, end_date: Time) -> Frame:
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
        tmp_df = calculate_costs_and_proceeds(name, group, end_date)
        result_df = pd.concat([result_df, tmp_df], ignore_index=True)

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


def calculate_all_portfolio_pnl(
    path: str, start_date: Time, end_date: Time, benchmark: str
) -> DictFrame:
    """
    Calculate the profit and loss (PnL) for all portfolios.

    Returns:
        A dictionary containing the calculated PnL for each portfolio.
    """
    result_dict = {}
    sheets = pd.ExcelFile(path).sheet_names

    for sheet in sheets:
        data = pd.read_excel(path, sheet_name=sheet)
        if len(data) == 0:
            print(f"Tab is empty for {sheet}")
            continue
        res = calculate_portfolio_pnl(data, end_date)
        res = res[(res["date"] >= start_date) & (res["date"] <= end_date)]
        group = res.groupby("ticker")["cumulative_quantity"].sum()
        tickers = group[group == 0].index
        res = res[~res["ticker"].isin(tickers)]
        result_dict[sheet] = res

    # add benchmark portfolio
    if benchmark != "":
        data = get_ticker_data(benchmark)
        data = data.loc[start_date:end_date][MARK_PRICE]
        benchmark_df = pd.DataFrame(
            {
                "date": data.index,
                "ticker": len(data) * [benchmark],
                "price": data.values,
            }
        )
        benchmark_df = benchmark_df[benchmark_df["date"].dt.is_month_end]
        benchmark_df["quantity"] = 1.0
        result_dict["Benchmark"] = calculate_portfolio_pnl(benchmark_df, end_date)

    return result_dict


def calculate_sharpe_ratio(ticker: str, end_date: Time) -> float:
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


def calculate_ytd(ticker: str, end_date: Time) -> Any:
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
    start = data.head(1)[MARK_PRICE].iloc[0]
    end = data.tail(1)[MARK_PRICE].iloc[0]
    ytd = ((end - start) / start) * 100
    return round(ytd, 2)
