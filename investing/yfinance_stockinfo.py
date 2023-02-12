import os
import datetime as dt
import warnings
from connect import connect
from myutils import sqlengine, sqlengine_pull_from_db, talk_to_me, timestamp, timestamp_onlyday, back_to_the_future
import pickle

import pandas as pd
import yfinance as yf

from tqdm import tqdm

from p5_get_sp500_list import save_sp500_tickers

from p1add_pricedata_to_database import push_df_to_db_replace, push_df_to_db_append, pull_df_from_db

pd.set_option("display.max.columns", None)


def get_stock_info(reload_sp500=False):
    """

    :param
    :param
    :return: info
    """
    if reload_sp500 is True:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    # tickers = ['DE', 'CVX', 'XOM']
    df_list = []
    for count, ticker in enumerate(tqdm(tickers)):
        data = yf.Ticker(ticker)
        # get stock info
        """ Ideas for building-up the dataframe"""
        try:
            info = data.info
            print(info)
        except Exception as e:
            warnings.warn(
                f'{ticker}: No summary info found, symbol may be delisted {e}',
                UserWarning)
            continue
        df = pd.DataFrame.from_dict(info, orient='index').transpose()
        df.reset_index(inplace=True)
        df['Symbol'] = ticker
        df.set_index('Symbol', inplace=True)
        #print(df)
        df_list.append(df)

        if count % 10 == 0:
            print(count)

    final_df = pd.concat(df_list, axis=0, join='outer')
    return final_df


def get_stock_fast_info(reload_sp500=False):
    """
    fast access to subset of stock info - how to access a lazy dictionary:
    :param
    :param
    :return: info
    """
    if reload_sp500 is True:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    tickers = ['DE', 'CVX', 'XOM']
    df_list = []
    for count, ticker in enumerate(tqdm(tickers)):
        data = yf.Ticker(ticker)
        # get stock basic info
        """ Ideas for building-up the dataframe"""
        fast_info = data.fast_info
        print(fast_info)
        print(fast_info['currency'])
        print(fast_info['timezone'])
        print(fast_info['regular_market_previous_close'])
        print(fast_info['fifty_day_average'])
        print(fast_info['two_hundred_day_average'])
        print(fast_info['ten_day_average_volume'])
        print(fast_info['three_month_average_volume'])
        print(fast_info['year_high'])
        print(fast_info['year_low'])
        print(fast_info['year_change'])

        if count % 10 == 0:
            print(count)


def get_stock_info_europe():
    """


    :return:
    """

    tickers = ['BAS.DE']
    df_list = []


if __name__ == '__main__':
    df = get_stock_info()
    push_df_to_db_replace(df, 'sp500_stockinfo')
    print(df)
    df_basic = get_stock_fast_info()
    #push_df_to_db_replace(df_basic, 'sp500_basicinfo')
    print(df_basic)
    #data = yf.Ticker('DE')
    #print(data.info)
    # get stock basic info
    """ Ideas for building-up the dataframe"""
    #fast_info = data.fast_info
    #print(fast_info)

    # show financials:
    # - income statement
    #print(data.income_stmt)
    #data.quarterly_income_stmt
    # - balance sheet
    #print(data.balance_sheet)
    #data.quarterly_balance_sheet
    # - cash flow statement
    #print(data.cashflow)
    #data.quarterly_cashflow
    # see `Ticker.get_income_stmt()` for more options


    # show news - you will only fetch the links to the media outlets - partly the news are not longer existent
    # after scraping
    #print(data.news)

