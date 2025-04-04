"""
Data sourcing functions.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import json
import re
import sys
from datetime import date
from time import sleep
from typing import Any, Dict

import pandas as pd
import requests
import yahooquery as yq
import yfinance as yf
from bs4 import BeautifulSoup

from src.report.errors import NoDataErr
from src.utils.types import Frame

ticker_data: Dict[str, Frame] = {}
ticker_metrics: Dict[str, Dict[str, Any]] = {}


def get_ticker_data(ticker: str) -> Frame:
    """
    Retrieve historical data for a given ticker symbol.

    Parameters:
    ticker (str): Ticker symbol for the desired ETF.

    Returns:
    Frame: DataFrame containing the historical data for the specified ticker.
    """
    if ticker in ticker_data.keys():
        return ticker_data[ticker]

    try:
        data = yf.download(ticker, progress=False, auto_adjust=True)
        if not len(data):
            print(f"No data from Yahoo finance for {ticker}")
            sys.exit()
    except Exception as e:
        print(f"Unable to get data from Yahoo finance for {ticker}: {e}")
        sys.exit()

    data.columns = data.columns.droplevel(1)
    data.index = pd.to_datetime(data.index).date
    data = data.reindex(pd.date_range(min(list(data.index)), date.today(), freq="D"))
    data = data.ffill()
    ticker_data[ticker] = data
    return data


def get_title_from_html(item: str) -> str:
    """
    Get title from HTML string.

    Parameters:
    item (str): HTML string

    Returns:
    str: title
    """
    title_match = re.search('title="([^"]+)"', item)
    if title_match:
        title = title_match.group(1)
        title = title.split("-", 1)[0].rstrip()
        return title
    else:
        return item


def get_anchor_from_html(item: str) -> str:
    """
    Get anchor tag from HTML string.

    Parameters:
    item (str): HTML string

    Returns:
    str: anchor tag
    """
    if len(item) <= 10:
        return item
    else:
        soup = BeautifulSoup(item, "html.parser")
        anchor_tag = soup.find("a")
        if anchor_tag:
            rel_attr = anchor_tag.get("rel")
            if rel_attr:
                return str(rel_attr[0])
    return ""


def get_etf_underlyings_external(ticker: str) -> Frame:
    """
    Extract underlying stock information for a list of ETF tickers from www.zacks.com, storing the data locally.

    Parameters:
    ticker (str): List of tickers for which to extract underlying stock information.

    Returns:
    Frame: DataFrame containing the extracted stock information, including ticker, stock symbol,
    company name, and weight.
    """
    url = f"https://www.zacks.com/funds/etf/{ticker}/holding"
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"
    }
    with requests.Session() as req:
        req.headers.update(header)
        r = req.get(url)
        html = r.text
        start = html.find("etf_holdings.formatted_data = ") + len(
            "etf_holdings.formatted_data = "
        )
        end = html.find(";", start)
        formatted_data = html[start:end].strip()
        try:
            data = json.loads(formatted_data)
        except Exception:
            raise NoDataErr("Unable to get underlyings data")

        symbols = [get_anchor_from_html(item[1]) for item in data]
        if len(symbols) == 1:
            raise NoDataErr("Unable to get underlyings data")

        names = [get_title_from_html(item[0]) for item in data]
        weights = [float(lst[3]) if lst[3] != "NA" else None for lst in data]

        if " bond" in get_ticker_info(ticker)["category"].lower():
            names = [i + " (Bond)" for i in names]

        df = pd.DataFrame({"Stock": symbols, "Company": names, "Weight": weights})
        df.insert(0, "ticker", ticker)
        df["Company"] = df["Company"].str.title()

    file_name = f"data/input/etf_underlyings/{ticker}.csv"
    df.to_csv(file_name, index=False)
    return df


def get_etf_underlyings_internal(ticker: str) -> Frame:
    """
    Extract underlying stock information for a list of ETF tickers local CSV files.

    Parameters:
    ticker (str): List of tickers for which to extract underlying stock information.

    Returns:
    Frame: DataFrame containing the extracted stock information, including ticker, stock symbol,
    company name, and weight.
    """
    file_name = f"data/input/etf_underlyings/{ticker}.csv"
    df = pd.read_csv(file_name)
    return df


def get_etf_underlyings(tickers: list[str], source: str) -> Frame:
    """
    Get ETF underlyings data from either external or internal source.

    Parameters:
    tickers (list[str]): List of tickers for which to extract underlying stock information.
    source (str): Source type.

    Returns:
    Frame: DataFrame containing the extracted stock information, including ticker, stock symbol,
    company name, and weight.
    """
    source_map = {
        "external": get_etf_underlyings_external,
        "internal": get_etf_underlyings_internal,
    }
    if source not in source_map:
        raise ValueError(f"Unknown source: {source}")

    res = [source_map[source](ticker) for ticker in tickers]
    res = pd.concat(res, ignore_index=True)
    return res


def get_ticker_metrics(ticker: str, retries: int = 5, delay: int = 1) -> Frame:
    """
    Fetch ticker metrics with retry mechanism.

    Parameters:
    ticker (str): Stock ticker symbol.
    retries (int): Max number of retries.
    delay (int): Delay (in seconds) between retries.

    Returns:
    Frame: Ticker data.
    """
    if ticker in ticker_metrics:
        return ticker_metrics[ticker]

    for attempt in range(retries):
        try:
            data = yq.Ticker(ticker).all_modules[ticker]
            if data == "Invalid Crumb":
                sleep(delay)
                continue
            ticker_metrics[ticker] = data
            return data
        except Exception as e:
            raise NoDataErr(f"Error fetching data for {ticker}: {e}")

    raise RuntimeError(f"Failed to get data for {ticker} after {retries} attempts.")


def get_metrics(tickers: list[str]) -> Frame:
    """
    Get ETF metrics from yahooquery.

    Parameters:
    tickers (list[str]): List of tickers for which get the metrics for.

    Returns:
    Frame: DataFrame containing the extracted metrics.
    """
    df_list = []
    for i in tickers:
        data = get_ticker_metrics(i)
        exp_ratio = data["fundProfile"]["feesExpensesInvestment"][
            "annualReportExpenseRatio"
        ]

        summary = data["summaryDetail"]
        pe_ratio = summary.get("trailingPE", 0.0)
        div_yield = summary.get("yield", 0.0)
        volume = summary["volume"]

        statistics = data["defaultKeyStatistics"]
        ytd = statistics.get("ytdReturn", 0.0)
        beta = statistics.get("beta3Year", 0.0)
        assets = statistics["totalAssets"]
        avg_return = statistics.get("threeYearAverageReturn", 0.0)

        statistics = data["fundPerformance"]["riskOverviewStatistics"]["riskStatistics"]
        sharpe = 0.0
        if len(statistics) > 0:
            sharpe = statistics[0]["sharpeRatio"]

        dict = {
            "Ticker": i,
            "Exp. Ratio": round(100 * exp_ratio, 2),
            "Div. Yield": round(100 * div_yield, 2),
            "Sharpe Ratio": sharpe,
            "Beta": beta,
            "PE Ratio": round(pe_ratio, 2),
            "Volume": volume,
            "Assets": round(assets / 1_000_000_000, 2),
            "YTD Return": round(100 * ytd, 2),
            "3yr Return": round(100 * avg_return, 2),
        }
        df = pd.DataFrame([dict])
        df_list.append(df)

    res = pd.concat(df_list, ignore_index=True)
    return res


def get_sector_weightings(tickers: list[str]) -> Frame:
    """
    Retrieve sector weightings for a list of tickers and returns them as a DataFrame.

    Parameters:
    tickers (list[str]): A list of stock ticker symbols.

    Returns:
    Frame: A DataFrame containing sector weightings for each ticker.
    """
    df_list = []
    for i in tickers:
        data = get_ticker_metrics(i)
        data = data["topHoldings"]["sectorWeightings"]
        df = pd.DataFrame(
            [(k, v) for d in data for k, v in d.items()], columns=["Sector", "Weight"]
        )
        df.insert(0, "Ticker", i)
        df_list.append(df)

    res = pd.concat(df_list, ignore_index=True)
    return res


def get_ticker_info(ticker: str) -> dict[str, str]:
    """
    Retrieve info data for a given ticker symbol.

    Parameters:
    ticker (str): Ticker symbol for the desired ETF.

    Returns:
    dict: Dictionary containing ticker name and description.
    """
    data = get_ticker_metrics(ticker)
    name = data["price"]["shortName"]
    category = data["fundProfile"]["categoryName"]
    description = data["summaryProfile"].get("longBusinessSummary", "NA")
    res = {"name": name, "category": category, "description": description}
    return res
