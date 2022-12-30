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

from tqdm import tqdm

yf.pdr_override()


def get_yahoo_pricedata(reload_sp500=False):
    '''
    Notes:
        - for that things won't get messy, we will create a directory for the pricedata with os
        - Yahoo Traceback still on 29th Dec 2022
        -     data = j["context"]["dispatcher"]["stores"]["HistoricalPriceStore"]
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
        - TypeError: string indices must be integers, not 'str'
        - use yfinance instead
    :param reload_sp500: False, since we already have a sp500tickers object on disc; if we want to reload from
    Wikipedia, then we call the function with the param True
    :return:
    '''
    if reload_sp500 is True:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)
            print(tickers)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    '''
    Variables:
    '''
    # start = dt.datetime(2022, 1, 1) #for Yahoo via pdr
    start = '2022-01-01'  # input('Startdatum im Format YYYY-MM-DD') usecase for yfinance
    end = dt.datetime.now()

    for ticker in tqdm(tickers):
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index('Date', inplace=True)
            # df = df.drop('Symbol', axis=1) in yfinance scheint es das symbol nicht zu geben
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


get_yahoo_pricedata(reload_sp500=False)

# Todo
'''
You will likely in time want to do some sort of force_data_update parameter to this function, since, right now, 
it will not re-pull data it already has. Since we're pulling daily data, you'd want to have this re-pulling 
at least the latest data. That said, if that's the case, you might be better off with using a database instead 
with a table per company, and then just pulling the most recent values from the Yahoo database. 
We'll keep things simple for now though!
'''
