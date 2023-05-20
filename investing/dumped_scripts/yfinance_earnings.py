"""
Stand April 2023: Yahoo lässt sich teilweise nicht mehr von yfinance scrapen...
Scripte funktionieren so nicht mehr
"""

import os
import datetime as dt
import time
import warnings
from investing.connect import connect
from investing.myutils import sqlengine, sqlengine_pull_from_db, talk_to_me, timestamp, timestamp_onlyday, back_to_the_future
import pickle

import pandas as pd
import yfinance as yf

from tqdm import tqdm

from investing.p5_get_sp500_list import save_sp500_tickers

from investing.p1add_pricedata_to_database import push_df_to_db_replace, push_df_to_db_append, pull_df_from_db

pd.set_option("display.max.columns", None)


def get_stock_earnings(reload_sp500=False):
    """
    Always create the yfinance Ticker-instance at first.
    Returns Dataframes of the last 4 years and the last 4 reported quarters.
    :param reload_sp500:
    :return:
    """
    if reload_sp500 is True:
        tickers = save_sp500_tickers()
    else:
        with open('../sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    tickers = ['DE'] #, 'CVX', 'XOM']
    df_list = []
    for count, ticker in enumerate(tqdm(tickers)):
        # create ticker instance:
        data = yf.Ticker(ticker)
        # get stock earnings
        earnings = data.earnings
        qearnings = data.quarterly_earnings

        rev_forecasts = data.revenue_forecasts
        earnings_forecasts = data.earnings_forecasts
        earnings_trend = data.earnings_trend

        # show next event (earnings, etc)
        calendar = data.calendar

        print(f'Earnings of {ticker}: {earnings}, Quarterly Earnings of {ticker}: {qearnings}')
        print(type(earnings))
        print(type(qearnings))

        print(f'Revenue-Forecasts of {ticker}: {rev_forecasts}, Earnings Forecasts of {ticker}: {earnings_forecasts}')
        print(f'Earnings-Trend of {ticker}: {earnings_trend}')
        print('')
        print(f'Calendar: {calendar}')

        if count % 10 == 0:
            print(count)


if __name__ == '__main__':
    get_stock_earnings()

