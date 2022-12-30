# import this

'''
In Windows muss man so die packages installieren
py -m pip install pandas
'''
import datetime as dt
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr

import yfinance as yf
yf.pdr_override()

'''
Konfigurationen
'''
style.use('ggplot')
# wird obsolet, wenn wir mit yf overriden - nothing will happen:
pd.set_option("display.max.columns", None)

'''
Variablen:
'''
start = '2022-01-01' # input('Startdatum im Format YYYY-MM-DD')
end = dt.datetime.now()
tickers = ['CMCL', 'GC=F', 'SPY', 'DE', 'CVX']
tickers_yf = 'CMCL' # yfinance nutzt nur einen string - auch bei mehreren Tickern ohne Komma

global dataframe


def get_pricedata(symbols: list, startdate: str, enddate: dt.datetime) -> str:
    '''
    Tipp: am Besten für mehrere Ticker nutzen
    die Syntax entspricht dem Pandas-Datareader, die API wurde aber von yfinance overridden ("hijacked")
    :param symbols: strings oder Liste (oder Liste von Listen oder Liste von strings, depends on API)
    :param startdate: string, wenn wir yahoo oder yfinance als Package nehmen
    :param enddate: aufpassen, ob wir dt.datetime oder Funktion timestamp aus myutils nehmen!
    :return: ein Pandas-Dataframe - ein Panel bei mehreren Tickern
    '''

    global dataframe
    try:
        # download panel data
        dataframe = pdr.get_data_yahoo(symbols, start=startdate, end=enddate)
    except Exception as e:
        print('type error: ', str(e))
        time.sleep(5)
        pass
    return dataframe


def get_symbol_returns_from_yahoo(symbol: str, startdate=None, enddate=None):
    '''

    Wrapper for pandos.io.data.gat_data_yahoo() but overridden by yf.
    Retrieve prices for symbols from yahoo and computes returns based on adjusted closing prices

    :param symbol: str,
        Symbol name to load, e.g. 'SPY'
    :param startdate: pandas.Timestamp compatible, optional
        Start date of time period to retrieve
    :param enddate: pandas.Timestamp compatible, optional
        End date of time period to retrieve
    :return: pandas.DataFrame
        Returns of symbol in requested period.
    '''

    try:
        px = pdr.get_data_yahoo(symbol, start=startdate, end=enddate)
        # was macht das pd.to_datetime??? Das funzt nicht...
        '''
        Pandas to_datetime() method helps to convert string Date time into Python Date time object.
        # making data frame from csv file
        data = pd.read_csv("todatetime.csv")
        # overwriting data after changing format
        data["Date"]= pd.to_datetime(data["Date"])
        # info of data
        data.info()
        # display
        data
        '''
        #px['Date'] = pd.to_datetime(px['Date'])
        #px.set_index('Date', drop=False, inplace=True)
        returns = px[['Adj Close']].pct_change().dropna()
    except Exception as e:
        warnings.warn(
            'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
            UserWarning)
        px = yf.download(symbol, start=startdate, end=enddate)
        returns = px[['Adj Close']].pct_change().dropna()

    returns.index = returns.index.tz_localize("UTC")
    returns.columns = [symbol]
    return returns


def get_pricedata_yfinance(ticker: str, startdate: str, enddate: dt.datetime):
    '''
    Tipp: am besten nur für einen Ticker nutzen
    data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = "SPY AAPL MSFT",

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "ytd",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "5d",

        # Whether to ignore timezone when aligning ticker data from
        # different timezones. Default is True. False may be useful for
        # minute/hourly data.
        ignore_tz = False,

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # attempt repair of missing data or currency mixups e.g. $/cents
        repair = False,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )
    :param ticker:
    :param startdate:
    :param enddate:
    :return:
    '''

    # fetching data for multiple tickers:
    data = yf.download(ticker, start=startdate, end=enddate)

    return data


def stock_info(symbol: str):
    '''
    Nutzt einige Features von yfinance, die als Listen oder dicts returned werden
    :param symbol: any ticker - direkt als string in Funktion eintragen oder oben Variable deklarieren
    :return: object, info, calendar, earnings, news
    '''
    obj = yf.Ticker(symbol)

    # get stock info
    info = obj.info

    # show next event (earnings, etc)
    calendar = obj.calendar

    # Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
    # Note: If more are needed use obj.get_earnings_dates(limit=XX) with increased limit argument.
    earnings = obj.earnings_dates

    # show news
    news = obj.news

    # get historical market data
    hist = obj.history(period="1y")

    # show meta information about the history (requires history() to be called first)
    hist_metadata = obj.history_metadata

    return obj, info, calendar, earnings, news, hist, hist_metadata


#df = get_pricedata(tickers, start, end)
#print(df)

#df_yf = get_pricedata_yfinance(tickers_yf, start, end)
#print(df_yf)
#print(df_yf.index)

#print(stock_info('DE'))

#df = get_symbol_returns_from_yahoo(tickers_yf, start, end)
#print(df)
