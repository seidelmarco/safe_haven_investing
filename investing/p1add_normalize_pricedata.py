import os
from urllib.request import Request, urlopen
import json
import ssl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pandas_datareader as pdr
from pandas.api.types import CategoricalDtype
from tqdm import tqdm
import warnings

import time
import datetime as dt
from myutils import timestamp

import yfinance as yf
yf.pdr_override()  # <== that's all it takes :-)

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

#newest yahoo API
import yfinance as yahoo_finance


pd.set_option("display.max.columns", None)

pd.core.common.is_list_like = pd.api.types.is_list_like

# make pandas to print dataframes nicely
pd.set_option('expand_frame_repr', False)


# ___variables___
tickers = ['DE']

START_DATE = '2022-01-01' # input('Startdatum im Format YYYY-MM-DD:')
END_DATE = '2022-11-01' # input('Enddatum im Format YYYY-MM-DD:')

startdate = '2022-01-01'  # input('Startdatum im Format YYYY-MM-DD') usecase for yfinance
enddate = dt.datetime.now()


def get_data(tickers, startdate, enddate):
    for count, ticker in enumerate(tqdm(tickers)):
        try:
            ticker_df = pdr.get_data_yahoo(ticker, startdate, enddate)
        except Exception as e:
            warnings.warn(
                'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
                UserWarning)
            # fetching data for multiple tickers:
            ticker_df = yf.download(ticker, start=startdate, end=enddate)
    # use numerical integer index instead of date
    ticker_df = ticker_df.reset_index()
    print(ticker_df)
    return ticker_df


def normalize_data(df):
    """
    df on input should contain only one column with the price data (plus dataframe index)
    :param df:
    :return: y will be a new column in a dataframe - we will call it 'norm' like so:
    df['norm'] = normalize_data(df['Adj Close'])
    """
    min = df.min()
    max = df.max()
    x = df
    # time series normalization part
    # y will be a column in a dataframe - we will call it 'norm'
    y = (x - min) / (max - min)
    return y


def normalize_pricedata(df_1col) -> float:
    """
    df on input should contain only one column with the price data (plus dataframe index)
    :param df_1col:
    :return: y will be a new column in a dataframe - we will call it 'norm' like so:
    df['norm'] = normalize_data(df['Adj Close'])
    """
    min = df_1col.min()
    max = df_1col.max()
    x = df_1col
    # time series normalization part
    y = (x - min) / (max - min)
    return y


if __name__ == '__main__':
    df = get_data(tickers, startdate, enddate)
    df['norm'] = normalize_data(df['Adj Close'])

    print(df)

    '''
    AHHHH Facepalm: wenn ich shift nehme, bekomme ich die täglichen returns, also immer mal + oder -,
    damit kann man keine Linie plotten, für Linien normalized Data nehmen
    '''

    # plot price
    plt.figure(figsize=(15, 5))
    plt.plot(df['Date'], df['Adj Close'])
    plt.title(f'Price chart (Adj Close) {tickers[0]}')
    plt.legend([tickers[0]], loc='upper left')

    #for i in ax.patches:
        #ax.annotate()
    plt.show()

    # plot normalized price chart
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(15, 5))
    plt.title(f'Normalized price chart {tickers[0]}')
    plt.plot(df['Date'], df['norm'], label=tickers[0])
    plt.legend([tickers[0]], loc='upper left')

    plt.show()

    ticker1 = ['CMCL']
    df1 = get_data(ticker1, startdate, enddate)
    df1['norm'] = normalize_data(df1['Adj Close'])

    ticker2 = ['IMPUY']
    df2 = get_data(ticker2, startdate, enddate)
    df2['norm'] = normalize_data(df2['Adj Close'])

    ticker3 = ['XOM']
    df3 = get_data(ticker3, startdate, enddate)
    df3['norm'] = normalize_data(df3['Adj Close'])

    ticker4 = ['GC=F']
    df4 = get_data(ticker4, startdate, enddate)
    df4['norm'] = normalize_data(df4['Adj Close'])

    # Extra Plot wegen Übersichtlichkeit
    ticker5 = ['GDX']
    df5 = get_data(ticker5, startdate, enddate)
    df5['norm'] = normalize_data(df5['Adj Close'])

    ticker6 = ['SPY']
    df6 = get_data(ticker6, startdate, enddate)
    df6['norm'] = normalize_data(df6['Adj Close'])

    # plot normalized price chart
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(15, 5))
    plt.title(f'Normalized price chart  {tickers[0]}  {ticker1[0]}  {ticker2[0]}  {ticker3[0]}  {ticker4[0]}')
    plt.plot(df['Date'], df['norm'])
    plt.plot(df1['Date'], df1['norm'])
    plt.plot(df2['Date'], df2['norm'])
    plt.plot(df3['Date'], df3['norm'])
    plt.plot(df4['Date'], df4['norm'])
    plt.legend([tickers[0], ticker1[0], ticker2[0], ticker3[0], ticker4[0]], loc='upper left')

    plt.show()

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(15, 5))
    plt.title(f'Normalized price chart {ticker5[0]}  {ticker6[0]}')
    # plt.plot(df['Date'], df['norm'])
    plt.plot(df5['Date'], df5['norm'])
    # plt.plot(df2['Date'], df2['norm'])
    # plt.plot(df3['Date'], df3['norm'])
    plt.plot(df6['Date'], df6['norm'])
    plt.legend([ticker5[0], ticker6[0]], loc='upper left')

    plt.show()

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(15, 5))
    plt.title(f'Normalized price chart {tickers[0]}  {ticker6[0]}')

    plt.plot(df['Date'], df['norm'])
    plt.plot(df6['Date'], df6['norm'])
    plt.legend([tickers[0], ticker6[0]], loc='upper left')

    plt.show()