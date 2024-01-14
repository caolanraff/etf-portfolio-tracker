# ETF Portfolio Tracker
The ETF portfolio tracker allows the tracking and analysis of multiple ETF portfolios, simply defined within an Excel file.
It has two main execution options:
1) Summary mode
2) Report mode

For summary mode, pass in the timeframe and the optional start/end dates.
The script will automatically work out the MTD and YTD timeframes if no start/end date provided.
This will return such information as the AUM, the portfolio returns in order of highest to lowest, and two charts.
One chart for the overall PnL return per portfolio, and one for the specific timeframe PnL per portfolio.

For report mode, an additional parameter 'report' should be passed in.
This will create a PDF report (sample below) with the following pages:
- Title page
  - Includes AUM and optional image
- Summary returns
  - With optional notes for best/worst
- Charts from the summary report
- New trades made in that timeframe
- Best and worst ETF performers
- ETF weightings
- Combined ETF weightings
- ETF metrics
  - Sharpe Ratio, Beta, Expense Ratio, PE Ratio, Yield, YTD returns
  - With optional highlighing for high/low values
- Highest weighted underlyings
- ETF percentage overlaps
- ETF Descriptions

There is also an option to add a benchmark ticker to compare portfolios against.
It will assume a purchase of 1 share at the end of every month.

### Samples
Sample input file: [portfolios.xlsx](data/input/portfolios.xlsx)

Sample output file: [advanced_report.pdf](data/output/advanced_report.pdf)

### Getting started
Clone the project and install poetry:
```
$ brew install poetry
$ poetry install
```

### Usage
```
$ poetry run python src/main.py --help
usage: main.py [-h] [--timeframe TIMEFRAME] [--start START] [--end END] [--config CONFIG] [--report]

optional arguments:
  -h, --help              show this help message and exit
  --timeframe TIMEFRAME   timeframe [MTD|YTD|adhoc]
  --start START           start date [YYYY-MM-DD]
  --end END               end date [YYYY-MM-DD]
  --config CONFIG         config [config/*.ini]
  --report                generate PDF report
```

##### Examples
```
poetry run python src/main.py --timeframe YTD

poetry run python src/main.py --timeframe MTD --start 2023-05-01 --end 2023-05-30

poetry run python src/main.py --timeframe MTD --config config/advanced.ini --report
```

### Configuration
There is a default.ini configuration file in the data/input directory, which will be used by default.
Multiple configs can be created and be passed in on an adhoc basis to create different reports.
The config settings are:
- Input
  - file --> excel file with the trades for each portfolio across different tabs
  - benchmark --> benchmark ticker to compare portfolios against
- TitlePage
  - title --> title on the title page
  - image --> image to be displayed on the title page
- SummaryPage
  - best --> comments to be made about the best portfolio
  - worst --> comments to be made about the worst portfolio
- WeightingsPage
  - other --> for the combined weights, if ETF weighting is below this value it will be moved into 'other' section
- MetricsPage
  - threshold --> list of thresholds for the outlined metrics
  - operator --> the operator to check against (.e.g >, <, =)
  - highlight --> colour of highlighting
- HoldingsPage
  - num_of_companies --> the number of companies to include in the top underlyings
  - threshold --> threshold for maximum % of holding in single company
- Output
  - file --> the name of the output file
