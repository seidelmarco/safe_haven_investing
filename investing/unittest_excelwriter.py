import unittest
import os
import warnings
from openpyxl import writer
import openpyxl
import pandas as pd
import pandas_datareader.data as pdr

from tqdm import tqdm
import datetime as dt

import yfinance as yf
yf.pdr_override()


def print_excel(path: str):
    """
    Probably since update to Pandas 1.5.3 the code raises a value error in case of df created by
    yfinance with index='Date' that there won't be written any excel-files:
    https://stackoverflow.com/questions/61802080/excelwriter-valueerror-excel-does-not-support-datetime-with-timezone-when-saving
    :param df:
    :param path: 'sp500_dfs/sp500_lastday_' + ohlc_attr + '.xlsx'
    :return: df
    """
    startdate = '2022-01-01'  # input('Startdatum im Format YYYY-MM-DD') usecase for yfinance
    enddate = dt.datetime.now()

    # main_df = pd.DataFrame()
    ticker = ['DE']

    try:
        df = pdr.get_data_yahoo(ticker, startdate, enddate)
    except Exception as e:
        warnings.warn(
            'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
            UserWarning)
        # fetching data for multiple tickers:
        df = yf.download(ticker, start=startdate, end=enddate)

    if not os.path.exists('tests'):
        os.makedirs('tests')

    print(df)

    # Todo debugging starts here:

    # check index with timezones
    print(df.index.dtype) # result: datetime64[ns, America/New_York]

    # Remove timezone from columns
    df.reset_index(inplace=True)
    print(df.dtypes)
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.set_index('Date', inplace=True)
    print(df.index.dtype)  # result: datetime64[ns] :-) ............ it works!

    # Example: df.to_excel('sp500_dfs/sp500_lastday_' + ohlc_attr + '.xlsx', engine='openpyxl')
    pd.DataFrame.to_excel(df, path, engine='openpyxl')

    return df


if __name__ == '__main__':
    print_excel(path='tests/test_print_excel.xlsx')


'''
class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
'''