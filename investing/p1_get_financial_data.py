# import this

"""
In Windows muss man so die packages installieren
py -m pip install pandas
"""
import sys
import datetime as dt
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr

from tqdm import tqdm
from myutils import push_df_to_db_replace, flatten

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
    """
    Tipp: am Besten für mehrere Ticker nutzen
    die Syntax entspricht dem Pandas-Datareader, die API wurde aber von yfinance overridden ("hijacked")
    :param symbols: strings oder Liste (oder Liste von Listen oder Liste von strings, depends on API)
    :param startdate: string, wenn wir yahoo oder yfinance als Package nehmen
    :param enddate: aufpassen, ob wir dt.datetime oder Funktion timestamp aus myutils nehmen!
    :return: ein Pandas-Dataframe - ein Panel bei mehreren Tickern
    """

    global dataframe
    try:
        # download panel data
        dataframe = pdr.get_data_yahoo(symbols, start=startdate, end=enddate)

    except Exception as e:
        warnings.warn(
            'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
            UserWarning)
        # fetching data for multiple tickers:
        dataframe = yf.download(symbols, start=startdate, end=enddate)

    finally:
        pass

    return dataframe


def get_symbol_returns_from_yahoo(symbol: str, startdate=None, enddate=None):
    """

    Wrapper for pandos.io.data.gat_data_yahoo() but overridden by yf.
    Retrieve prices for symbols from yahoo and computes returns based on adjusted closing prices

    :param symbol: str,
        Symbol name to load, e.g. 'SPY'
    :param startdate: pandas.Timestamp compatible, optional
        Start date of time period to retrieve - STRING!
    :param enddate: pandas.Timestamp compatible, optional
        End date of time period to retrieve - STRING!
    :return: pandas.DataFrame
        Returns of symbol in requested period.
    """

    try:
        df = pdr.get_data_yahoo(symbol, start=startdate, end=enddate)
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
        #df['Date'] = pd.to_datetime(df['Date'])
        #df.set_index('Date', drop=False, inplace=True)
        returns = df[['Adj Close']].pct_change().dropna()
    except Exception as e:
        warnings.warn(
            'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
            UserWarning)
        df = yf.download(symbol, start=startdate, end=enddate)
        returns = df[['Adj Close']].pct_change().dropna()

    returns.index = returns.index.tz_localize("UTC")
    returns.columns = [symbol]
    return returns


def get_pricedata_yfinance(ticker: str, startdate: str, enddate: dt.datetime):
    """
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
    """

    # fetching data for multiple tickers:
    data = yf.download(ticker, start=startdate, end=enddate)

    return data


def get_dividends(ticker):
    """

    :param ticker:
    :return:
    """
    data = yf.Ticker(ticker)
    divs = data.dividends
    return divs


def multi_tickers():
    tickers = yf.Tickers('msft aapl de')

    # access each ticker using (example)
    info = tickers.tickers['MSFT'].info
    history = tickers.tickers['AAPL'].history(period="1mo")
    actions = tickers.tickers['DE'].actions
    print(info)
    print(history)
    print(actions)


def get_currency_pairs():
    """

    :return: main_df with rates
    """
    # excel to dataframe und dann tickers-list....

    with open('sp500_dfs/currency_pairs.xlsx', 'rb') as f:
        tickers_col = pd.read_excel(f, header=0, engine='openpyxl', usecols=[1])
        tickers_list = tickers_col.values.tolist()
        flat_list = flatten(tickers_list)
        # flat_list = [item for sublist in tickers_list for item in sublist]

    print(flat_list)

    '''
    Variables:
    '''
    startdate = '2022-01-01'  # input('Startdatum im Format YYYY-MM-DD') usecase for yfinance
    enddate = dt.datetime.now()

    joined_frames_list = []

    for count, ticker in enumerate(tqdm(flat_list)):
        try:
            df = pdr.get_data_yahoo(ticker, startdate, enddate)
        except Exception as e:
            warnings.warn(
                f'Yahoo Finance read failed: {e}, falling back to YFinance',
                UserWarning)
            df = yf.download(ticker, start=startdate, end=enddate)

        df.rename(columns={'Adj Close': str(ticker)}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        #print(df)
        #break

        joined_frames_list.append(df)

    main_df = pd.concat(joined_frames_list, axis=1, join='outer')
    print(main_df)
    return main_df


if __name__ == '__main__':

    #print(get_pricedata(tickers, startdate=start, enddate=end))

    #print(get_dividends('DE'))

    #print(get_dividends('CTRA'))
    # multi_tickers()

    # print(get_symbol_returns_from_yahoo('BAS.DE', startdate='2023-01-01', enddate='2023-02-13'))

    get_currency_pairs()
    push_df_to_db_replace(get_currency_pairs(), 'pricedata_curreny_pairs')
    sys.stdout.write('Alle Währungspaare gezogen und auf Datanbank geschaufelt.')
