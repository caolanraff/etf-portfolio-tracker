# portfolio-tracker
Python Portfolio Tracker

### Usage
```
$ python3 portfolio_tracker.py --help
usage: portfolio_tracker.py [-h] [--timeframe TIMEFRAME] [--start START] [--end END] [--report]

optional arguments:
  -h, --help            show this help message and exit
  --timeframe TIMEFRAME
                        timeframe [MTD|YTD|adhoc]
  --start START         start date [YYYY-MM-DD]
  --end END             end date [YYYY-MM-DD]
  --report              generate PDF report
```

### Examples
```
python3 portfolio_tracker.py --timeframe YTD

python3 portfolio_tracker.py --timeframe MTD --start 2023-05-01 --end 2023-05-23

python3 portfolio_tracker.py --timeframe MTD --start 2023-05-01 --end 2023-05-23 --report
```