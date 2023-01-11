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
import tqdm

import time
from datetime import datetime
from myutils import timestamp

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

#newest yahoo API
import yfinance as yahoo_finance

import yfinance as yf
yf.pdr_override()  # <== that's all it takes :-)

pd.set_option("display.max.columns", None)

pd.core.common.is_list_like = pd.api.types.is_list_like

# make pandas to print dataframes nicely
pd.set_option('expand_frame_repr', False)


# ___variables___
ticker = 'DE'

START_DATE = '2022-01-01' # input('Startdatum im Format YYYY-MM-DD:')
END_DATE = '2022-11-01' # input('Enddatum im Format YYYY-MM-DD:')

start_time = START_DATE
end_time = datetime.today()


def get_data(ticker, start_time, end_time):
    connected = False
    while not connected:
        try:
            ticker_df = pdr.get_data_yahoo(ticker, start_time, end_time)
            connected = True
            print('Connected to Yahoo')
        except Exception as e:
            print('type error: ', str(e))
            time.sleep(5)
            pass
    # use numerical integer index instead of date
    ticker_df = ticker_df.reset_index()
    print(ticker_df)
    return ticker_df


df = get_data(ticker, start_time, end_time)


def normalize_data(df):
    '''
    df on input should contain only one column with the price data (plus dataframe index)
    :param df:
    :return:
    '''
    min = df.min()
    max = df.max()
    x = df
    # time series normalization part
    # y will be a column in a dataframe - we will call it 'norm'
    y = (x - min) / (max - min)
    return y


df['norm'] = normalize_data(df['Adj Close'])

print(df)

'''
AHHHH Facepalm: wenn ich shift nehme, bekomme ich die täglichen returns, also immer mal + oder -,
damit kann man keine Linie plotten, für Linien normalized Data nehmen
'''

# plot price
plt.figure(figsize=(15,5))
plt.plot(df['Date'], df['Adj Close'])
plt.title('Price chart (Adj Close) ' + ticker)
plt.show()

# plot normalized price chart
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,5))
plt.title('Normalized price chart ' + ticker)
plt.plot(df['Date'], df['norm'])

plt.show()


ticker1='CMCL'
df1 = get_data(ticker1, start_time, end_time)
df1['norm'] = normalize_data(df1['Adj Close'])

ticker2='IMPUY'
df2 = get_data(ticker2, start_time, end_time)
df2['norm'] = normalize_data(df2['Adj Close'])

ticker3='XOM'
df3 = get_data(ticker3, start_time, end_time)
df3['norm'] = normalize_data(df3['Adj Close'])

ticker4='GC=F'
df4 = get_data(ticker4, start_time, end_time)
df4['norm'] = normalize_data(df4['Adj Close'])

# Extra Plot wegen Übersichtlichkeit
ticker5='GDX'
df5 = get_data(ticker5, start_time, end_time)
df5['norm'] = normalize_data(df5['Adj Close'])

ticker6='SPY'
df6 = get_data(ticker6, start_time, end_time)
df6['norm'] = normalize_data(df6['Adj Close'])


# plot normalized price chart
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,5))
plt.title('Normalized price chart ' + ticker + ' ' + ticker1 + ' ' + ticker2 + ' ' + ticker3 + ' ' + ticker4)
#plt.plot(df['Date'], df['norm'])
plt.plot(df1['Date'], df1['norm'])
#plt.plot(df2['Date'], df2['norm'])
#plt.plot(df3['Date'], df3['norm'])
plt.plot(df4['Date'], df4['norm'])

plt.show()


plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,5))
plt.title('Normalized price chart ' + ticker5 + ' ' + ticker6)
#plt.plot(df['Date'], df['norm'])
plt.plot(df5['Date'], df5['norm'])
#plt.plot(df2['Date'], df2['norm'])
#plt.plot(df3['Date'], df3['norm'])
plt.plot(df6['Date'], df6['norm'])

plt.show()


plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,5))
plt.title('Normalized price chart ' + ticker + ' ' + ticker6)

plt.plot(df['Date'], df['norm'])
plt.plot(df6['Date'], df6['norm'])

plt.show()


def normalize_pricedata(df_1col) -> float:
    """
    df on input should contain only one column with the price data (plus dataframe index)
    :param df_1col:
    :return: y will be a new column in a dataframe - we will call it 'norm'
    """
    min = df_1col.min()
    max = df_1col.max()
    x = df_1col
    # time series normalization part
    y = (x - min) / (max - min)
    return y
