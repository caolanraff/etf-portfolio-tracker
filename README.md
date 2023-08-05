# ETF Portfolio Tracker
The ETF portfolio tracker allows the tracking and analysis of multiple portfolios, simply defined within an Excel file.
It has two main execution options:
1) Summary mode
2) Report mode

For summary mode, simply pass in the timeframe and the optional start/end dates.
The script will automatically work out the MTD and YTD timeframes if no start/end date provided.
This will return such information as the AUM, the portfolio returns in order of highest to lowest and two charts.
One chart for the overall PnL return per portfolio, and one for the specific timeframe PnL per portfolio.

For report mode, an additional parameter 'report' should be passed in.
This will create a PDF report (example located in data/output) with the following pages:
- Title page
  - Includes AUM and optional image
- Summary returns per portfolio
  - With optional notes for best/worst
- Charts from the summary report
- New trades made in that timeframe
- Best and worst ETF performers per portfolio
- ETF weightings
- Combined ETF weightings
- ETF metrics
  - Beta, Expense Ratio, PE Ratio, Yield, YTD returns
  - With optional highlighing for high/low values
- Highest weighted underlyings per portfolio
- ETF percentage overlaps per portfolio

### Getting started
Clone the project and then within the project run:
```
pip install -r requirements.txt
```

### Usage
```
$ python3 portfolio_tracker.py --help
usage: portfolio_tracker.py [-h] [--timeframe TIMEFRAME] [--start START] [--end END] [--config CONFIG] [--report]

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
python3 portfolio_tracker.py --timeframe YTD

python3 portfolio_tracker.py --timeframe MTD --start 2023-05-01 --end 2023-05-30

python3 portfolio_tracker.py --timeframe MTD --config config/advanced.ini --report
```

### Configuration
There is a default.ini configuration file in the data/input directory, which will be used by default.
Multiple configs can be created and be passed in on an adhoc basis to create different reports.
The config settings are:
- Input
  - file --> excel file with the trades for each portfolio across different tabs
  - image --> image to be displayed on the title page
- Text
  - title --> title on the title page
  - best --> comments to be made about the best portfolio
  - worst --> comments to be made about the worst portfolio
- Weightings
  - other --> for the combined weights, if ETF weighting is below this value it will be moved into 'other' section
- Metrics
  - threshold --> list of thresholds for the outlined metrics
  - operator --> the operator to check again (.e.g >, <, =)
  - highlight --> colour of highlighting
- Holdings
  - top --> the number of companies to include in the top holdings
- Output
  - file --> the name of the output file
