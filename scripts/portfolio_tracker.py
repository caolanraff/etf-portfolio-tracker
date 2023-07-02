import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
from yahoo_fin import stock_info as si
from matplotlib.backends.backend_pdf import PdfPages
import pdfrw
import os
from fpdf import FPDF
import math
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('--timeframe', type=str, help='timeframe [MTD|YTD|adhoc]')
parser.add_argument('--start', default='', type=str, help='start date [YYYY-MM-DD]')
parser.add_argument('--end', default='', type=str, help='end date [YYYY-MM-DD]')
parser.add_argument('--report', action='store_true', help='generate PDF report')
parser.add_argument('--path', default='./', type=str, help='directory path')
parser.add_argument('--config', default='config/config.ini', type=str, help='config file')
args = parser.parse_args()

timeframe = args.timeframe
start_date = args.start
end_date = args.end
now = datetime.now()
if start_date == '':
    if timeframe == 'MTD':
        start_date = datetime(now.year, now.month, 1)
    elif timeframe == 'YTD':
        start_date = datetime(now.year, 1, 1)
    else:
        print('[ERROR] Unknown timeframe')
        exit()
else:
    start_date = datetime.strptime(start_date, '%Y-%m-%d')

if end_date == '':
    end_date = datetime(now.year, now.month, now.day-1)
else:
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

config = configparser.ConfigParser()
config.read(args.path + '/' + args.config)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


### calculate pnl
def calculate_portfolio_pnl(file_path, sheet):
    df = pd.read_excel(file_path, sheet_name=sheet)
    if len(df) == 0:
        print("[WARN] Tab is empty for " + sheet)
        return

    df['date'] = pd.to_datetime(df['date'])
    max_date = pd.Timestamp.now() - pd.Timedelta(days=1)
    grouped = df.groupby('ticker')
    result_df = pd.DataFrame()

    # Calculate running cumulative quantity and running average entry price per ticker
    for name, group in grouped:
        # Create a date range from the earliest to latest date for this ticker
        date_range = pd.date_range(start=group['date'].min(), end=max_date, freq='D')
        # Create a new dataframe with the complete date range and the ticker symbol
        date_df = pd.DataFrame({'date': date_range, 'ticker': name})
        # Merge the original dataframe with the new dataframe
        merged_df = pd.merge(date_df, group, on=['date', 'ticker'], how='outer')
        # Fill missing values with 0
        merged_df = merged_df.fillna(0)
        # Calculate running cumulative quantity and running average entry price
        merged_df['cumulative_quantity'] = merged_df['quantity'].cumsum()
        merged_df['total_cost'] = merged_df['quantity'] * merged_df['price']
        merged_df['cumulative_cost'] = merged_df['total_cost'].cumsum()
        merged_df['average_entry_price'] = merged_df['cumulative_cost'] / merged_df['cumulative_quantity']
        # Append the resulting dataframe for this ticker to the global dataframe
        result_df = pd.concat([result_df, merged_df], ignore_index=True)

    # Download the adjusted close price from Yahoo Finance for each ticker and date
    min_date = result_df['date'].min()
    min_date = min_date - timedelta(days=7)
    prices = {}
    for ticker in result_df['ticker'].unique():
        ticker_data = yf.Ticker(ticker).history(start=min_date, end=max_date)
        ticker_data = ticker_data.reindex(pd.date_range(min_date, max_date, freq='D'))
        ticker_data['Close'] = ticker_data['Close'].interpolate()
        prices[ticker] = ticker_data['Close']

    # Merge the price data onto the original dataframe
    result_df['market_price'] = result_df.apply(lambda row: prices[row['ticker']].loc[pd.Timestamp(row['date'])], axis=1)
    # Calculate the notional value based on the market price
    result_df['notional_value'] = result_df['cumulative_quantity'] * result_df['market_price']
    # Calculate PnL
    result_df['pnl'] = result_df['cumulative_quantity'] * (result_df['market_price'] - result_df['average_entry_price'])
    # Sort the dataframe by date
    result_df = result_df.sort_values('date')
    # Calculate the PNL per date
    result_df['portfolio_pnl'] = result_df.groupby('date')['pnl'].transform('sum')
    # Calculate the portfolio value per date
    result_df['portfolio_cost'] = result_df.groupby('date')['cumulative_cost'].transform('sum')
    result_df['portfolio_value'] = result_df.groupby('date')['notional_value'].transform('sum')
    # Calculate the PNL per date as a percentage
    result_df['pnl_pct_per_date'] = 100 * (result_df['portfolio_value'] - result_df['portfolio_cost']) / result_df['portfolio_cost']
    result_df = result_df.reset_index(drop=True)
    return result_df


### Calculate PnL
def calculate_all_portfolio_pnl():
    print('[INFO] Calculating portfolio PnLs')
    result_dict = {}
    file = args.path + '/data/input/' + config.get('Input', 'file')
    sheets = pd.ExcelFile(file).sheet_names
    for sheet in sheets:
        res = calculate_portfolio_pnl(file, sheet)
        if res is None:
            continue
        res = res[(res['date'] >= start_date) & (res['date'] <= end_date)]
        result_dict[sheet] = res
    return result_dict


### Save dataframe to PDF
def save_dataframe_to_pdf(df, file, highlight_columns=None, thresholds=None, operators=None):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', ha='left')
        else:
            cell.set_text_props(ha='left')
            if highlight_columns and thresholds and operators:
                for i, col_name in enumerate(highlight_columns):
                    try:
                        col_index = df.columns.get_loc(col_name)
                    except KeyError:
                        raise ValueError(f"Column '{col_name}' not found in dataframe")
                    if col == col_index:
                        cell_value = float(cell.get_text().get_text())
                        if operators[i] == ">" and cell_value > thresholds[i]:
                            cell.set_facecolor('red')
                        elif operators[i] == "<" and cell_value < thresholds[i]:
                            cell.set_facecolor('red')
    pp = PdfPages(file)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()


### Create title page
def create_title_page(aum):
    pdf_output = FPDF()
    pdf_output.add_page()
    title = config.get('Text', 'title')
    subtitle = end_date.strftime('%B %Y') + ' Meeting'
    aum = 'AUM: ' + aum
    pdf_output.set_font('Arial', 'B', 36)
    pdf_output.cell(0, 80, title, 0, 1, 'C')
    pdf_output.set_font('Arial', '', 24)
    pdf_output.cell(0, 20, subtitle, 0, 1, 'C')
    pdf_output.set_font('Arial', '', 16)
    pdf_output.cell(0, 20, aum, 0, 1, 'C')
    image = config.get('Input', 'image')
    if image != '':
        image_file = args.path + '/data/input/' + image
        pdf_output.image(image_file, x=55, y=150, w=100, h=100)
    pdf_output.output('title.pdf')


### Get AUM
def get_aum(result_dict):
    print('[INFO] Get AUM')
    portfolio_val = 0

    for name, df in result_dict.items():
        res = df.loc[df['date'] == end_date].iloc[0]
        portfolio_val += res['portfolio_value']

    aum = '$' + '{:,.0f}'.format(portfolio_val)
    return aum


### Get summary info
def get_summary(result_dict, save_to_file):
    print('[INFO] Get summary info')
    val = []

    for name, df in result_dict.items():
        som = df.loc[df['date'] == start_date].iloc[0]
        som = 100 * (som['portfolio_value'] - som['portfolio_cost']) / som['portfolio_cost']
        eom = df.loc[df['date'] == end_date].iloc[0]
        eom = 100 * (eom['portfolio_value'] - eom['portfolio_cost']) / eom['portfolio_cost']
        val.append(eom - som)

    summary = pd.DataFrame({'Investor': list(result_dict.keys()), timeframe:val})
    summary = summary.sort_values(by=timeframe, ascending=False)
    summary[timeframe] = summary[timeframe].round(3)
    summary = summary.reset_index(drop=True)
    summary.loc[0, 'Notes'] = config.get('Text', 'best')
    summary.loc[summary.index[-1], 'Notes'] = config.get('Text', 'worst')
    summary['Notes'] = summary['Notes'].fillna('')

    if save_to_file:
        save_dataframe_to_pdf(summary, "summary.pdf")
    else:
        print(summary)


### Chart
def plot_performance_charts(result_dict, save_to_file):
    print('[INFO] Plotting performance charts')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    handles = []  # handles for the legend
    labels = []   # labels for the legend

    for name, df in result_dict.items():
        line1, = ax1.plot(df['date'], df['pnl_pct_per_date'], label=name)
        if name not in labels:
            handles.append(line1)
            labels.append(name)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('PnL')
        ax1.set_title('Overall PnL Change')
        ax1.set_xlim(start_date, end_date)

        df = df.assign(pnl_change=df['pnl_pct_per_date'].diff().cumsum())
        line2, = ax2.plot(df['date'], df['pnl_change'], label=name)
        if name not in labels:
            handles.append(line2)
            labels.append(name)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('PnL')
        ax2.set_title(timeframe + ' PnL Change')
        ax2.set_xlim(start_date, end_date)

    for ax in (ax1, ax2):
        ax.tick_params(axis='x', labelrotation=20)
    fig.legend(handles, labels)

    if save_to_file:
        plt.savefig('performance.pdf')
    else:
        plt.show()


# new trades
def new_trades(result_dict):
    print('[INFO] Getting new trades')
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        trades = df.loc[df['quantity'] != 0]
        tickers = trades['ticker'].to_list()
        tickers = ', '.join(tickers)
        result_df = result_df.append({'Investor': key, 'Trades': tickers}, ignore_index=True)

    save_dataframe_to_pdf(result_df, "new_trades.pdf")


### best and worst
# @TODO: best and worst with timeframe
def best_and_worst(result_dict):
    print('[INFO] Getting best and worst ETFs')
    result_df = pd.DataFrame()

    for key, df in result_dict.items():
        df = df.loc[df['date'] == end_date].copy()
        total_notional = df['notional_value'].sum()
        df['pnl_val'] = df['pnl'] / total_notional * 100
        df['pnl_pct'] = df['pnl'] / df['notional_value'] * 100

        best_portfolio_contribution = df.loc[df['pnl_val'].idxmax(), 'ticker']
        worst_portfolio_contribution = df.loc[df['pnl_val'].idxmin(), 'ticker']
        best_pnl_pct = df.loc[df['pnl_pct'].idxmax(), 'ticker']
        worst_pnl_pct = df.loc[df['pnl_pct'].idxmin(), 'ticker']

        result_df = result_df.append({'Investor': key,
                                      'Best Portfolio Contributer': best_portfolio_contribution,
                                      'Worst Portfolio Contributer': worst_portfolio_contribution,
                                      'Best PnL %': best_pnl_pct,
                                      'Worst PnL %': worst_pnl_pct
                                      }, ignore_index=True)

    save_dataframe_to_pdf(result_df, "best_and_worst.pdf")


### pie chart
def plot_pie_charts(result_dict):
    print('[INFO] Plotting ETF weightings')
    n = len(result_dict)
    num_cols = 3
    num_rows = math.ceil(n / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
    if not isinstance(axs[0], np.ndarray):
        axs = [[axs[i * num_cols + j] for j in range(num_cols)] for i in range(num_rows)]
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    for i, (key, df) in enumerate(result_dict.items()):
        row = i // num_cols
        col = i % num_cols
        df = df.loc[df['date'] == end_date]
        axs[row][col].pie(df['notional_value'], labels=df['ticker'], autopct='%1.1f%%', radius=1.2)
        axs[row][col].set_title(key, y=1.1, fontdict={'fontsize': 14, 'fontweight': 'bold'})

    for i in range(n, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row][col])

    plt.suptitle('ETF Weightings')
    plt.savefig('weightings.pdf')


# Get metrics
def get_metrics(result_dict):
    print('[INFO] Getting metrics')
    result_df = pd.DataFrame()
    metrics = ['Beta (5Y Monthly)', 'Expense Ratio (net)', 'PE Ratio (TTM)', 'Yield', 'YTD Daily Total Return']

    for key, df in result_dict.items():
        tickers = list(df['ticker'].unique())
        for ticker in tickers:
            res = si.get_quote_table(ticker)
            my_dict = {k: v for k, v in res.items() if k in metrics}
            df = pd.DataFrame([my_dict])
            df['Investor'] = key
            df['Ticker'] = ticker
            result_df = result_df.append(df, ignore_index=True)

    result_df['Expense Ratio (net)'] = result_df['Expense Ratio (net)'].str.rstrip('%')
    result_df['Yield'] = result_df['Yield'].str.rstrip('%')
    result_df['YTD Daily Total Return'] = result_df['YTD Daily Total Return'].str.rstrip('%')
    result_df = result_df[['Investor', 'Ticker'] + metrics]

    save_dataframe_to_pdf(result_df, "metrics.pdf",
                          metrics,
                          [1.2, 0.5, 30, 0.7, -5],
                          [">", ">", ">", "<", "<"]
                          )


### Merge pdf files
def merge_pdfs(file_list, output_file):
    print('[INFO] Merging files')
    pdf_output = pdfrw.PdfWriter()
    for file_name in file_list:
        pdf_input = pdfrw.PdfReader(file_name)
        for page in pdf_input.pages:
            pdf_output.addpage(page)
        os.remove(file_name)
    pdf_output.write(output_file)


### main
def comp():
    print(f"[INFO] Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})")
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    print('[INFO] AUM: ' + aum)
    get_summary(res_dict, False)
    plot_performance_charts(res_dict, False)


def report():
    print(f"[INFO] Running report for {timeframe} ({start_date:%Y-%m-%d} - {end_date:%Y-%m-%d})")
    res_dict = calculate_all_portfolio_pnl()
    aum = get_aum(res_dict)
    create_title_page(aum)
    get_summary(res_dict, True)
    plot_performance_charts(res_dict, True)
    new_trades(res_dict)
    best_and_worst(res_dict)
    plot_pie_charts(res_dict)
    get_metrics(res_dict)
    merge_pdfs(['title.pdf', 'summary.pdf', 'performance.pdf', 'new_trades.pdf', 'best_and_worst.pdf', 'weightings.pdf', 'metrics.pdf'],
               args.path + '/data/output/' + config.get('Output', 'file'))


if __name__ == "__main__":
    if args.report:
        report()
    else:
        comp()
