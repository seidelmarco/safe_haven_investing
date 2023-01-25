import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import pandas_datareader.data as pdr

from p1_get_financial_data import get_pricedata_yfinance

#style.use('ggplot')
style.use('fivethirtyeight')

'''
Variablen:
'''
start = '2022-01-01' # input('Startdatum im Format YYYY-MM-DD')
end = dt.datetime.now()
tickers = ['CMCL', 'GC=F', 'SPY', 'DE', 'CVX']
tickers_yf = 'DE' # yfinance nutzt nur einen string - auch bei mehreren Tickern ohne Komma


def ohlc_candlestick(symbol: str, startdate: str, enddate: dt.datetime):
    """
    IMPORTANT! This function use the legacy method ohlc_candlestick from the deprecated mpl_finance:
    Example: from mplfinance.original_flavor import candlestick_ohlc (use this until you are accustomed
    to the methods of mplfinance
    For the future use the function "ohlc_candlestick_update" which uses the newer and maintained
    mplfinance-Package

    We need proper OHLC-Data. Be aware of splits then we can't use 'close' though. We resample a 10D-average,
    for that we can see the candles better. We normalize data. We also normalize the Volume, since it would be
    too granular otherwise.
    Different from the tutorial we won't need to reset the index, since yfinance already results a proper df.
    :param symbol: one ticker
    :param startdate: a string
    :param enddate: a dt.datetime, set to now/today
    :return:
    """

    df = get_pricedata_yfinance(symbol, startdate=startdate, enddate=enddate)

    # we need proper OHLC-Data
    df_ohlc = df['Adj Close'].resample('10D').ohlc()
    df_ohlc = df_ohlc.reset_index()
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    print(df_ohlc)

    df_volume = df['Volume'].resample('10D').sum()
    print(df_volume)

    fig = plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

    plt.show()

    # TODO wie wäre es mit Boxplot in Pandas? Nochmal nacharbeiten, wie ich das nutzen kann.


def ohlc_candlestick_update(symbol: str, startdate: str, enddate: dt.datetime):
    """
    This newer function uses the maintained package mplfinance 0.12.9b7

    :param symbol:
    :param startdate:
    :param enddate:
    :return:
    """

    df = get_pricedata_yfinance(symbol, startdate=startdate, enddate=enddate)
    df.index.name = 'Datum'
    #df.rename(columns={'Volume': 'volume'}, inplace=True)
    print(df)

    # you hardly can see the candles, thus we resample and normalize to 10 day-intervals
    df = df['Adj Close'].resample('10D').ohlc()
    mpf.plot(df, type='candle', mav=(4, 9))
    #mpf.plot(df, type='ohlc')
    plt.show()


ohlc_candlestick('CMCL', start, end)

ohlc_candlestick_update('CMCL', start, end)


