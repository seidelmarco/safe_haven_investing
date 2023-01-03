import os
from connect import connect
from myutils import sqlengine, talk_to_me
import pickle

import datetime as dt
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


def get_yahoo_sp500_ohlc(reload_sp500=False):
    '''
    Ich muss mir die Funktion aus p6 ableiten und selber bauen, weil ich ja täglich neu laden will, auch wenn
    ich den Ticker schon habe
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
        df.rename(columns={'Adj Close': ticker}, inplace=True)

        # später umbenannt wieder hinzufügen
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        #print(df)

        '''
        nächste Schritte: ohlc-Daten und extra Funktion für ausgesuchte Ticker wie CMCL
        '''

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    main_df.to_excel('sp500_dfs/sp500_joined_closes.xlsx', engine='openpyxl')
    return main_df


sp500_df = get_yahoo_sp500_ohlc(reload_sp500=False)
print(sp500_df)


def push_df_to_db(df):
    '''
    You can use talk_to_me() or connect()
    :return:
    '''

    # talk_to_me()

    connect()

    engine = sqlengine()

    df.to_sql('sp500_adjclose', con=engine, if_exists='replace', )


push_df_to_db(sp500_df)
