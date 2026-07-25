"""Converts files downloaded from etfdb.com and puts them into the required format for the report."""

import argparse
import csv
import re
from pathlib import Path


def convert_etf_holdings(input_file: str, output_dir: str) -> Path:
    """
    Convert ETF holdings CSV from detailed format to simplified format.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory where output file should be saved
    """
    input_path = Path(input_file).expanduser()
    output_path = Path(output_dir).expanduser()

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract ETF ticker from the input filename or file content
    etf_ticker = None
    holdings = []

    with open(input_path, "r", encoding="utf-8") as f:
        # Read through the file to find the ticker and holdings data
        lines = f.readlines()

        # Try to extract ticker from first line (e.g., "HACK: Amplify Cybersecurity ETF")
        first_line = lines[0].strip()
        ticker_match = re.match(r"^([A-Z]+):", first_line)
        if ticker_match:
            etf_ticker = ticker_match.group(1)

        if not etf_ticker:
            raise ValueError("Could not extract ETF ticker from file")

        # Find where the holdings data starts
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Holding,Symbol,Weighting"):
                header_idx = i
                break

        if header_idx is None:
            raise ValueError("Could not find holdings data header")

        # Parse holdings data
        reader = csv.DictReader(lines[header_idx:])
        for row in reader:
            if row["Symbol"] and row["Weighting"]:
                # Clean up the weighting (remove % sign)
                weight = row["Weighting"].strip().rstrip("%")
                holdings.append(
                    {
                        "ticker": etf_ticker,
                        "symbol": row["Symbol"].strip(),
                        "company": row["Holding"].strip().strip('"'),
                        "weight": weight,
                    }
                )

    # Create output filename
    output_file = output_path / f"{etf_ticker}.csv"

    # Write output file
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "Stock", "Company", "Weight"])

        for holding in holdings:
            writer.writerow(
                [
                    holding["ticker"],
                    holding["symbol"],
                    holding["company"],
                    holding["weight"],
                ]
            )

    print(f"Converted {input_path} to {output_file}")
    print(f"Processed {len(holdings)} holdings")
    return output_file


def main() -> int:
    """Parse CLI arguments and convert an ETF holdings CSV to simplified format."""
    parser = argparse.ArgumentParser(
        description="Convert ETF holdings CSV from detailed format to simplified format"
    )
    parser.add_argument(
        "input_file",
        help="Path to input CSV file (e.g., ~/Downloads/HACK-holdings.csv)",
    )
    parser.add_argument(
        "output_dir", help="Directory where output file should be saved"
    )

    args = parser.parse_args()

    try:
        convert_etf_holdings(args.input_file, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
