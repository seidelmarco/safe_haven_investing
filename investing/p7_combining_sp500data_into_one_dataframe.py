import bs4 as bs
import pickle
import requests

import time
import datetime as dt
import os  # os is to check for, and create, directories
import warnings

import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf

from p5_get_sp500_list import save_sp500_tickers
from p6_get_sp500_pricedata import get_yahoo_pricedata

from tqdm import tqdm

from openpyxl import writer
import openpyxl

yf.pdr_override()

# save_sp500_tickers()
# get_yahoo_pricedata(reload_sp500=False)


def compile_data():
    '''
    For the start we just compile the Adj Close column.

    - new symbol GEHC added by running this function
    :return:
    '''
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        '''
        You do not need to use Python's enumerate here, I am just using it so we know where we are in the process of 
        reading in all of the data. You could just iterate over the tickers. From this point, we *could* 
        generate extra columns with interesting data, like:
        '''
        #df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        #df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']

        # wir nennen die Spalte Adj Close 'Ticker' damit wir die 503 Einträge unterscheiden können
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df)
    main_df.to_csv('sp500_joined_closes.csv')
    main_df.to_excel('sp500_joined_closes.xlsx',  engine='openpyxl')


compile_data()

# Todo: RuntimeWarning: invalid value encountered in cast
#   values = values.astype(str)


