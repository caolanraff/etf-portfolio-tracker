"""
Data sourcing functions.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import json
import re
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.utils import initcap


def get_title_from_html(item: str) -> str:
    """
    Get title from HTML string.

    Args:
        item: HTML string
    Returns:
        title
    """
    title_match = re.search('title="([^"]+)"', item)
    if title_match:
        return title_match.group(1).split("-", 1)[0]
    else:
        return item


def get_anchor_from_html(item: str) -> str:
    """
    Get anchor tag from HTML string.

    Args:
        item: HTML string
    Returns:
        anchor tag
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


def get_etf_underlyings(tickers: List[str]) -> pd.DataFrame:
    """
    Extract underlying stock information for a list of ETF tickers.

    Args:
        tickers: List of tickers for which to extract underlying stock information.
    Returns:
        DataFrame containing the extracted stock information, including ticker, stock symbol,
        company name, and weight.
    """
    df_list = []
    for ticker in tickers:
        url = f"https://www.zacks.com/funds/etf/{ticker}/holding"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"
        }
        with requests.Session() as req:
            req.headers.update(headers)  # type: ignore
            r = req.get(url)  # type: ignore
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
                continue

            symbols = [get_anchor_from_html(item[1]) for item in data]
            names = [get_title_from_html(item[0]) for item in data]
            weights = [float(lst[3]) if lst[3] != "NA" else None for lst in data]

            df = pd.DataFrame({"Stock": symbols, "Company": names, "Weight": weights})
            df.insert(0, "ticker", ticker)
            df["Company"] = df["Company"].apply(initcap)
            df_list.append(df)

    result_df = pd.concat(df_list, ignore_index=True)
    return result_df


def get_yahoo_quote_table(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Scrapes data elements from Yahoo Finance's quote page for a given ticker.

    Args:
        ticker: Ticker symbol of the desired ETF.
    Returns:
        Dictionary containing scraped data elements with attribute-value pairs.
    """
    url = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker
    try:
        tables = pd.read_html(
            requests.get(url, headers={"User-agent": "Mozilla/5.0"}).text
        )
    except Exception as e:
        print(f"Unable to get metrics from Yahoo finance for {ticker}: {e}")
    data = pd.concat([tables[0], tables[1]])
    data.columns = ["attribute", "value"]
    data = data.sort_values("attribute")
    data = data.drop_duplicates().reset_index(drop=True)
    result = {key: val for key, val in zip(data.attribute, data.value)}
    return result
