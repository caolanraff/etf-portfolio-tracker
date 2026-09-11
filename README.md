# ETF Portfolio Tracker

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Track and analyse multiple ETF portfolios defined in a simple Excel file — get a quick console summary, or generate a polished, multi-page PDF report.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Samples](#samples)
- [License](#license)

## Overview

Each portfolio is defined as its own tab in an Excel workbook, listing the trades made over time. The tracker reads that file and computes cost basis, realised/unrealised PnL, and portfolio value from the raw trades, then presents the results in one of two modes:

1. **Summary mode** — a quick console overview.
2. **Report mode** — a full PDF report, suitable for sharing.

An optional benchmark ticker can be configured to compare portfolio performance against — modelled as a monthly purchase of one share, so it reflects a comparable dollar-cost-averaging strategy rather than a single lump-sum buy-and-hold.

## Features

**Summary mode** prints:
- Assets Under Management (AUM)
- Portfolio returns, ranked highest to lowest
- Two charts: overall PnL per portfolio, and PnL over the selected timeframe

**Report mode** generates a PDF with:

| Page | Contents |
| --- | --- |
| Title page | Report title, AUM, and an optional logo image |
| Table of contents | Auto-generated, with page numbers for every section |
| Summary | Portfolio returns and alpha vs. benchmark, with optional best/worst notes |
| Performance charts | Overall and timeframe PnL, per portfolio |
| New trades | Buys and sells made in the period |
| Best & worst performers | Top/bottom mover per portfolio, by price and by PnL contribution |
| Best & worst performers (combined) | Top/bottom movers across all portfolios |
| ETF weightings | Individual and combined portfolio composition |
| Metrics | Sharpe ratio, beta, expense ratio, PE ratio, yield, YTD/3yr return — with optional highlighting for outlier values |
| Risk metrics | Volatility, Sharpe ratio, and max drawdown over a trailing 1-year window (or since inception, if younger) |
| Top holdings | Highest-weighted underlying companies |
| Sector weightings | Portfolio exposure by sector |
| ETF overlaps | Percentage overlap between ETFs in a portfolio |
| ETF descriptions | Fund descriptions for every ETF held |

Every page is numbered, and the merged PDF opens straight to a clickable-feeling table of contents.

## Getting Started

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/)

### Installation

```bash
brew install poetry
poetry install
```

## Usage

### CLI reference

```bash
poetry run python -m src.cli.main --help
```

```
usage: main.py [-h] [--timeframe TIMEFRAME] [--start START] [--end END]
               [--report] [--path PATH] [--config CONFIG]

options:
  -h, --help              show this help message and exit
  --timeframe TIMEFRAME   timeframe [MTD|YTD|adhoc]
  --start START           start date [YYYY-MM-DD]
  --end END               end date [YYYY-MM-DD]
  --report                generate a PDF report (defaults to a console summary)
  --path PATH             directory containing data/input and data/output (default: current directory)
  --config CONFIG         config file [config/*.ini] (default: config/default.ini)
```

If `--start`/`--end` are omitted, they're worked out automatically from `--timeframe` (start of month for MTD, start of year for YTD).

### Examples

```bash
# Console summary, year-to-date
poetry run python -m src.cli.main --timeframe YTD

# Console summary, a specific date range
poetry run python -m src.cli.main --timeframe MTD --start 2023-05-01 --end 2023-05-30

# Full PDF report, using a specific config
poetry run python -m src.cli.main --timeframe MTD --config config/advanced.ini --report
```

## Configuration

`config/default.ini` is used if `--config` isn't passed. Create additional config files to produce different reports on an ad hoc basis.

| Section | Key | Description |
| --- | --- | --- |
| `Input` | `file` | Excel file with the trades for each portfolio, one tab per portfolio |
| | `benchmark` | Benchmark ticker to compare portfolios against |
| `TitlePage` | `title` | Title shown on the title page |
| | `image` | Optional image to display on the title page |
| `SummaryPage` | `best` | Comment attached to the best-performing portfolio |
| | `worst` | Comment attached to the worst-performing portfolio |
| `WeightingsPage` | `other` | For the combined weighting chart, ETFs below this weight are grouped into "Other" |
| `MetricsPage` | `threshold` | Threshold values used to highlight metrics outside the norm |
| | `operator` | Comparison operator per threshold (e.g. `>`, `<`, `=`) |
| | `highlight` | Highlight colour for flagged cells |
| `HoldingsPage` | `source` | `external` to scrape underlying holdings live, or `internal` to read a previously cached copy — always run `external` first |
| | `num_of_companies` | Number of companies to include in the top-holdings page |
| | `threshold` | Maximum allowed weight (%) in a single underlying company before it's flagged |
| `Output` | `file` | Output PDF filename |

## Samples

- Sample input file: [portfolios.xlsx](data/input/portfolios.xlsx)
- Sample output report: [advanced_report.pdf](data/output/advanced_report.pdf)

## License

[MIT](LICENSE.txt)
