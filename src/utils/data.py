"""
Data sourcing functions.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import json
import re
import sys
from datetime import date
from typing import Dict

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from src.utils.types import Frame

ticker_data: Dict[str, pd.DataFrame] = {}
ticker_info: Dict[str, pd.DataFrame] = {}
underlying_data: Dict[str, pd.DataFrame] = {}


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
        data = yf.download(ticker, progress=False)
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


def get_ticker_info(ticker: str) -> Frame:
    """
    Retrieve info data for a given ticker symbol.

    Parameters:
    ticker (str): Ticker symbol for the desired ETF.

    Returns:
    Frame: DataFrame containing the data info for the specified ticker.
    """
    if ticker in ticker_info.keys():
        return ticker_info[ticker]

    try:
        data = yf.Ticker(ticker).info
    except Exception as e:
        print(f"Unable to get data from Yahoo finance for {ticker}: {e}")
        sys.exit()

    ticker_info[ticker] = data
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


def get_etf_underlyings(ticker: str) -> Frame:
    """
    Extract underlying stock information for a list of ETF tickers.

    Parameters:
    ticker (str): List of tickers for which to extract underlying stock information.

    Returns:
    Frame: DataFrame containing the extracted stock information, including ticker, stock symbol,
    company name, and weight.
    """
    if ticker in underlying_data.keys():
        return underlying_data[ticker]

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
        except Exception as e:
            print(f"Unable to get underlyings for {ticker}: {e}")
            return pd.DataFrame()

        symbols = [get_anchor_from_html(item[1]) for item in data]
        names = [get_title_from_html(item[0]) for item in data]
        weights = [float(lst[3]) if lst[3] != "NA" else None for lst in data]

        if " bond" in get_ticker_info(ticker)["category"].lower():
            names = [i + " (Bond)" for i in names]

        df = pd.DataFrame({"Stock": symbols, "Company": names, "Weight": weights})
        df.insert(0, "ticker", ticker)
        df["Company"] = df["Company"].str.title()

    underlying_data[ticker] = df
    return df
