import os
from connect import connect
from myutils import sqlengine, talk_to_me, timestamp, timestamp_onlyday, back_to_the_future
import pickle

import datetime as dt
from datetime import timedelta
import warnings

from openpyxl import writer
import openpyxl
import pandas as pd
import pandas_datareader.data as pdr

from tqdm import tqdm

from p5_get_sp500_list import save_sp500_tickers

import yfinance as yf
yf.pdr_override()


'''
get yahoo pricedata for all 500 SP Tickers
'''


def get_yahoo_sp500_adjclose(reload_sp500=False):
    '''
    Since postgres-tables are limited to 1,600 columns, you only can use 3 columns per sp500-ticker. So, here we
    just grab adusted close

    Rowsize is 8160 Bytes max, so we can only use one column per ticker
    :return: df
    '''
    if reload_sp500 is True:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)
            print(tickers)

    if not os.path.exists('sp500_dfs'):
        os.makedirs('sp500_dfs')

    '''
    Variables:
    '''
    # start = dt.datetime(2022, 1, 1) #for Yahoo via pdr
    startdate = '2022-01-01'  # input('Startdatum im Format YYYY-MM-DD') usecase for yfinance
    enddate = dt.datetime.now()

    main_df = pd.DataFrame()

    # only for testing, for speed and performance reasons just test with one ticker-symbol:
    # tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY', 'MTNOY']
    for count, ticker in enumerate(tqdm(tickers)):
        # just in case your connection breaks, we'd like to save our progress!
        try:
            df = pdr.get_data_yahoo(ticker, startdate, enddate)
        except Exception as e:
            warnings.warn(
                'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
                UserWarning)
            # fetching data for multiple tickers:
            df = yf.download(ticker, start=startdate, end=enddate)
        #print(df)
        # an dieser Stelle brauchen wir keinen index setzen, weil 'Date' schon der Index ist df = data.set_index('Date', inplace=True)
        #print(df.index)

        # df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        # df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']

        # wir nennen die Spalte Adj Close 'Ticker' damit wir die 503 Einträge unterscheiden können
        df.rename(columns={'Adj Close': ticker+'_Adj_Close'}, inplace=True)

        # später umbenannt wieder hinzufügen
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        # print(df)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    main_df.to_excel('sp500_dfs/sp500_adjclose.xlsx', engine='openpyxl')
    return main_df


def get_sp500_ohlc_today(reload_sp500=False):
    '''
    Just update the table sp500_ohlc daily with one row

    Since postgres-tables are limited to 1,600 columns, you only can use 3 columns per sp500-ticker. So, here we
    just grab adusted close

    Rowsize is 8160 Bytes max, so we can only use one column per ticker
    :return: df
    '''
    if reload_sp500 is True:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)
            print(tickers)

    if not os.path.exists('sp500_dfs'):
        os.makedirs('sp500_dfs')

    '''
    Variables:
    '''
    # start = dt.datetime(2022, 1, 1) #for Yahoo via pdr
    backone_string = back_to_the_future()
    print(backone_string)
    startdate = backone_string
    enddate = None

    main_df = pd.DataFrame()

    # only for testing, for speed and performance reasons just test with one ticker-symbol:
    # tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY']
    for count, ticker in enumerate(tqdm(tickers)):
        # just in case your connection breaks, we'd like to save our progress!
        try:
            df = pdr.get_data_yahoo(ticker, startdate, enddate)
        except Exception as e:
            warnings.warn(
                'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
                UserWarning)
            # fetching data for multiple tickers:
            df = yf.download(ticker, start=startdate, end=enddate)
        #print(df)
        # an dieser Stelle brauchen wir keinen index setzen, weil 'Date' schon der Index ist df = data.set_index('Date', inplace=True)
        #print(df.index)

        # df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        # df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']

        # wir nennen die Spalte Adj Close 'Ticker' damit wir die 503 Einträge unterscheiden können
        df.rename(columns={'Adj Close': ticker+'_Adj_Close'}, inplace=True)

        # später umbenannt wieder hinzufügen
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        # print(df)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    main_df.to_excel('sp500_dfs/sp500_adjclose_only_daily.xlsx', engine='openpyxl')
    return main_df


def get_yahoo_ohlc_selection():
    '''
    I need crucially my own ticker list which is reliable due to the reasons, that there are too many unforeseen
    changes in the SP500 (like the new ticker GEHC on 2023-01-04) and I need securities which are not constituents
    of the SP500
    :param ticker:
    :param startdate:
    :param enddate:
    :return: main_df
    '''
    tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY', 'MTNOY', 'BAS.DE', 'MUV2.DE']
    print(tickers)

    if not os.path.exists('sp500_dfs'):
        os.makedirs('sp500_dfs')

    '''
    Variables:
    '''
    # start = dt.datetime(2022, 1, 1) #for Yahoo via pdr
    startdate = '2022-01-01'  # input('Startdatum im Format YYYY-MM-DD') usecase for yfinance
    enddate = dt.datetime.now()

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tqdm(tickers)):
        # just in case your connection breaks, we'd like to save our progress!
        try:
            df = pdr.get_data_yahoo(ticker, startdate, enddate)
        except Exception as e:
            warnings.warn(
                'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
                UserWarning)
            # fetching data for multiple tickers:
            df = yf.download(ticker, start=startdate, end=enddate)
        #print(df)
        # an dieser Stelle brauchen wir keinen index setzen, weil 'Date' schon der Index ist df = data.set_index('Date', inplace=True)
        #print(df.index)

        df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']

        # wir nennen die Spalte Adj Close 'Ticker' damit wir die 503 Einträge unterscheiden können
        df.rename(columns={'Adj Close': ticker+'_Adj_Close',
                           'Open': ticker+'_Adj_Close',
                           'High': ticker+'_High',
                           'Low': ticker+'_Low',
                           'Close': ticker+'_Close',
                           'Volume': ticker+'_Volume'}, inplace=True)

        # später umbenannt wieder hinzufügen
        # df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        # print(df)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    main_df.to_excel('sp500_dfs/portfolio_selection_ohlc.xlsx', engine='openpyxl')
    return main_df


def push_df_to_db(df, tablename: str):
    '''
    You can use talk_to_me() or connect()
    :tablename: str
    :return:
    '''

    # talk_to_me()

    connect()

    engine = sqlengine()

    df.to_sql(tablename, con=engine, if_exists='append', chunksize=100)


def pull_df_from_db():
    connect()
    engine = sqlengine()
    sql = ''
    df = pd.read_sql(sql, con=engine)
    return df


# sp500_df = get_yahoo_sp500_adjclose(reload_sp500=False)
# print(sp500_df)

# push_df_to_db(sp500_df, tablename='sp500_adjclose')


# sp500_only1day = get_sp500_ohlc_today(reload_sp500=False)
# print(sp500_only1day)

# push_df_to_db(sp500_only1day, tablename='sp500_ohlc')

selection = get_yahoo_ohlc_selection()
print(selection)

push_df_to_db(selection, tablename='selection_ohlc')
