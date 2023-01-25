import os
import sys
from connect import connect
from myutils import sqlengine, sqlengine_pull_from_db, talk_to_me, timestamp, timestamp_onlyday, back_to_the_future
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


def get_yahoo_sp500_ohlc(ohlc_attr: str = 'adjclose', reload_sp500=False):
    """
    This function grabs one column at a time from the ohlc-data and puts it into one dataframe per type
    :param ohlc_attr: beim callen entscheiden lassen, welche Spalte nicht gedropped wird
    dann nur open, high usw. in df und main df schreiben lassen
    :param reload_sp500:
    Spalte wieder auf nur ticker umbenennen
    :return:
    """

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

    # only for testing, for speed and performance reasons just investing with one ticker-symbol:
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
        # print(df)
        # an dieser Stelle brauchen wir keinen index setzen, weil 'Date' schon der Index ist df = data.set_index('Date', inplace=True)
        # print(df.index)

        match ohlc_attr:
            case 'open':
                df.rename(columns={'Open': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            case 'high':
                df.rename(columns={'High': ticker}, inplace=True)
                df.drop(['Adj Close', 'Open', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            case 'low':
                df.rename(columns={'Low': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Open', 'Close', 'Volume'], axis=1, inplace=True)
            case 'close':
                df.rename(columns={'Close': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)
            case 'volume':
                df.rename(columns={'Volume': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Low', 'Close', 'Open'], axis=1, inplace=True)
            case default:
                # wir nennen die Spalte Adj Close 'Ticker' damit wir die 503 Einträge unterscheiden können
                df.rename(columns={'Adj Close': ticker}, inplace=True)
                df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    main_df.to_excel('sp500_dfs/sp500_' + ohlc_attr + '.xlsx', engine='openpyxl')
    return main_df


def get_yahoo_sp500_adjclose(reload_sp500=False):
    """
    Since postgres-tables are limited to 1,600 columns, you only can use 3 columns per sp500-ticker. So, here we
    just grab adusted close

    Rowsize is 8160 Bytes max, so we can only use one column per ticker
    :return: df
    """
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

    # only for testing, for speed and performance reasons just investing with one ticker-symbol:
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
        df.rename(columns={'Adj Close': ticker}, inplace=True)

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


def get_sp500_ohlc_today(ohlc_attr: str = 'adjclose', reload_sp500=False):
    """
    Just update the table sp500_ohlc daily with one row

    Since postgres-tables are limited to 1,600 columns, you only can use 3 columns per sp500-ticker. So, here we
    just grab adusted close

    Rowsize is 8160 Bytes max, so we can only use one column per ticker
    :return: df
    """
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

    # only for testing, for speed and performance reasons just investing with one ticker-symbol:
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

        match ohlc_attr:
            case 'open':
                df.rename(columns={'Open': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            case 'high':
                df.rename(columns={'High': ticker}, inplace=True)
                df.drop(['Adj Close', 'Open', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            case 'low':
                df.rename(columns={'Low': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Open', 'Close', 'Volume'], axis=1, inplace=True)
            case 'close':
                df.rename(columns={'Close': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)
            case 'volume':
                df.rename(columns={'Volume': ticker}, inplace=True)
                df.drop(['Adj Close', 'High', 'Low', 'Close', 'Open'], axis=1, inplace=True)
            case default:
                df.rename(columns={'Adj Close': ticker}, inplace=True)
                df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    # Todo in all requests: https://stackoverflow.com/questions/61802080/
    #  excelwriter-valueerror-excel-does-not-support-datetime-with-timezone-when-saving

    main_df.reset_index(inplace=True)
    main_df['Date'] = main_df['Date'].dt.tz_localize(None)
    main_df.set_index('Date', inplace=True)
    # print(main_df.index)

    main_df.to_excel('sp500_dfs/sp500_lastday_' + ohlc_attr + '.xlsx', engine='openpyxl')
    return main_df


def get_yahoo_ohlc_selection():
    """
    I need crucially my own ticker list which is reliable due to the reasons, that there are too many unforeseen
    changes in the SP500 (like the new ticker GEHC on 2023-01-04) and I need securities which are not constituents
    of the SP500
    :param ticker:
    :param startdate:
    :param enddate:
    :return: main_df
    """
    tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY', 'MTNOY', 'BAS.DE', 'MUV2.DE', 'BLDP', 'KO', 'DLTR',
               'XOM', 'JNJ', 'KHC', 'MKC', 'MSFT', 'NEL.OL', 'OGN', 'SKT', 'TDG', 'DRD']
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


def get_selection_ohlc_today():
    '''
    Just update the table selection_ohlc daily with one row

    Since postgres-tables are limited to 1,600 columns, you only can use 3 columns per sp500-ticker.

    Beware: Rowsize is 8160 Bytes max
    :param ticker:
    :param startdate:
    :param enddate:
    :return: main_df
    '''

    tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY', 'MTNOY', 'BAS.DE', 'MUV2.DE', 'BLDP', 'KO', 'DLTR',
               'XOM', 'JNJ', 'KHC', 'MKC', 'MSFT', 'NEL.OL', 'OGN', 'SKT', 'TDG']
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

    # only for testing, for speed and performance reasons just investing with one ticker-symbol:
    # tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY']
    for count, ticker in enumerate(tqdm(tickers)):
        # just in case your connection breaks, we'd like to save our progress!
        try:
            '''
            data = yf.download(  # or pdr.get_data_yahoo(...
                # tickers list or string as well
                tickers="SPY AAPL MSFT",

                # use "period" instead of start/end
                # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                # (optional, default is '1mo')
                period="ytd",
            '''
            df = pdr.get_data_yahoo(ticker, period="1d")
        except Exception as e:
            warnings.warn(
                'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
                UserWarning)
            # fetching data for multiple tickers:
            df = yf.download(ticker, start=startdate, end=enddate)
        # print(df)
        # an dieser Stelle brauchen wir keinen index setzen, weil 'Date' schon der Index ist df = data.set_index('Date', inplace=True)
        # print(df.index)

        df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']

        # wir nennen die Spalte Adj Close 'Ticker' damit wir die 503 Einträge unterscheiden können
        df.rename(columns={'Adj Close': ticker + '_Adj_Close',
                           'Open': ticker + '_Adj_Close',
                           'High': ticker + '_High',
                           'Low': ticker + '_Low',
                           'Close': ticker + '_Close',
                           'Volume': ticker + '_Volume'}, inplace=True)

        # später umbenannt wieder hinzufügen
        # df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        # print(df)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    main_df.to_excel('sp500_dfs/portfolio_selection_ohlc_only_daily.xlsx', engine='openpyxl')
    return main_df


def get_yahoo_ohlc_commodities():
    """
    I need crucially my own ticker list for continuous commodities,
    :param ohlc_attr:
    :param ticker:
    :param startdate:
    :param enddate:
    :return: main_df
    """
    tickers = ['GC=F', 'PL=F', 'HG=F', 'CL=F']
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
        df.rename(columns={'Adj Close': ticker + '_Adj_Close',
                           'Open': ticker + '_Open',
                           'High': ticker + '_High',
                           'Low': ticker + '_Low',
                           'Close': ticker + '_Close',
                           'Volume': ticker + '_Volume'}, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    main_df.to_excel('sp500_dfs/commodities_ohlc.xlsx', engine='openpyxl')
    return main_df


def push_df_to_db_append(df, tablename: str):
    """
    You can use talk_to_me() or connect()
    :tablename: str
    :return:
    """

    # talk_to_me()

    connect()

    engine = sqlengine()

    # Todo: how to inherit if_exists to push_df_to_db-function?
    df.to_sql(tablename, con=engine, if_exists='append', chunksize=100)


def push_df_to_db_replace(df, tablename: str):
    """
    You can use talk_to_me() or connect()
    :tablename: str
    :return:
    """

    # talk_to_me()

    connect()

    engine = sqlengine()

    # Todo: how to inherit if_exists to push_df_to_db-function?
    df.to_sql(tablename, con=engine, if_exists='replace', chunksize=100)


def pull_df_from_db(sql='sp500_adjclose'):
    """
    # Todo : Funktion umschreiben, dass ich aus allen Tabellen und ausgewählte Spalten beim callen wählen kann
    :sql: default is table 'sp500_adjclose' - you can pass any table from the db as argument
    :return:
    """
    connect()
    engine = sqlengine_pull_from_db()

    # an extra integer-index-column is added
    # df = pd.read_sql(sql, con=engine)
    # Column 'Date' is used as index
    df = pd.read_sql(sql, con=engine, index_col='Date', parse_dates=['Date'], )
    '''
    #columns=['CMCL_Adj_Close',
                                                                                      # 'MTNOY_Adj_Close',
                                                                                      # 'GC=F_Adj_Close'])
    columns=['XOM_Adj_Close', 'AAPL_Adj_Close', 'DE_Adj_Close', 'BKNG_Adj_Close']
    '''
    return df


if __name__ == '__main__':
    ohlc_attr_input = input('OHLC-Attribut: ')
    # Todo: here still build in sysvar for command line prompt via calling the file without IDE
    sp500_df_lastday_per_attr = get_sp500_ohlc_today(ohlc_attr=ohlc_attr_input, reload_sp500=False)
    push_df_to_db_append(sp500_df_lastday_per_attr, tablename='sp500_'+ohlc_attr_input)

    #sp500_df_per_attr = get_yahoo_sp500_ohlc(ohlc_attr=ohlc_attr_input, reload_sp500=False)

    #push_df_to_db(sp500_df_per_attr, tablename='sp500_'+ohlc_attr_input)

    df = pull_df_from_db(sql='sp500_'+ohlc_attr_input)
    print(df)

    #sp500_df_per_attr_lastday = get_sp500_ohlc_today(ohlc_attr=ohlc_attr_input, reload_sp500=False)
    #push_df_to_db(sp500_df_per_attr_lastday, tablename='sp500_' + ohlc_attr_input)

    #df = pull_df_from_db(sql='sp500_' + ohlc_attr_input)
    #print(df)

    #commodities_ohlc = get_yahoo_ohlc_commodities()
    #push_df_to_db(commodities_ohlc, tablename='commodities_ohlc')
    #print(commodities_ohlc)

    #sp500_volume = get_yahoo_sp500_ohlc()
    #print(sp500_volume)

    #sp500_df = get_yahoo_sp500_adjclose(reload_sp500=False)
    #print(sp500_df)

    #push_df_to_db(sp500_df, tablename='sp500_adjclose')


    #sp500_only1day = get_sp500_ohlc_today(reload_sp500=False)
    #print(sp500_only1day)

    #push_df_to_db(sp500_only1day, tablename='sp500_adjclose')

    #selection = get_yahoo_ohlc_selection()
    #print(selection)

    #push_df_to_db(selection, tablename='selection_ohlc')


    # dailysel = get_selection_ohlc_today()
    # print(dailysel)

    # push_df_to_db(dailysel, tablename='selection_ohlc')

    #df = pull_df_from_db(sql='sp500_'+ohlc_attr_input)
    #print(df)

    #sys.exit('Jetzt ist aber Schluss hier!')

