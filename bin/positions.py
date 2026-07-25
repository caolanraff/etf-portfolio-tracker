"""
ETF Position & Average Price Calculator.

Loads a trades Excel file and prints the current net position and
weighted-average entry price for each ETF, per portfolio.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.cli.const import STOCK_SPLITS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def process_stock_splits(df: pd.DataFrame, stock_splits: pd.DataFrame) -> pd.DataFrame:
    """Apply stock splits to trades (same logic as calcs.py)."""
    if stock_splits.empty:
        return df

    unique_tickers = set(df["ticker"].tolist())
    split_tickers = set(stock_splits["ticker"].tolist())
    if not unique_tickers.intersection(split_tickers):
        return df

    for _, row in stock_splits.iterrows():
        mask = (df["ticker"] == row["ticker"]) & (df["date"] <= row["date"])
        df.loc[mask, "quantity"] *= row["ratio"]
        df.loc[mask, "price"] /= row["ratio"]
    return df


def positions_for_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Return net position and weighted-average buy price per ticker."""
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    trades = process_stock_splits(trades, pd.DataFrame(STOCK_SPLITS))

    position = trades.groupby("ticker")["quantity"].sum()

    buys = trades[trades["quantity"] > 0].copy()
    buys["notional"] = buys["quantity"] * buys["price"]
    grouped = buys.groupby("ticker").agg(
        total_qty=("quantity", "sum"),
        total_cost=("notional", "sum"),
    )
    grouped["avg_price"] = grouped["total_cost"] / grouped["total_qty"]

    result = pd.DataFrame({"position": position})
    result = result.join(grouped[["avg_price"]], how="left")
    result = result[result["position"] != 0]
    return result.sort_index()


def main() -> None:
    """Print open positions for each portfolio sheet in a trades Excel file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to the trades Excel file")
    args = parser.parse_args()

    sheets = pd.ExcelFile(args.file).sheet_names

    for sheet in sheets:
        data = pd.read_excel(args.file, sheet_name=sheet)
        if len(data) == 0:
            print(f"Tab is empty for {sheet}")
            continue

        print("=" * 60)
        print(f"Portfolio: {sheet}")
        print("=" * 60)
        res = positions_for_trades(data)
        if res.empty:
            print("  (no open positions)")
        else:
            print(res.to_string(float_format=lambda x: f"{x:,.4f}"))
        print()


if __name__ == "__main__":
    main()
