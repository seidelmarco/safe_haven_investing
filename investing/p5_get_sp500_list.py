import re

import bs4 as bs
import pickle

import pandas as pd
import requests

import datetime as dt

from investing.myutils import connect, sqlengine

import time


def push_df_to_db_replace_beautifulsoup(df, tablename: str):
    """
    from investing.p1add_pricedata_to_database import push_df_to_db_replace # does not work:
    ImportError: (most likely due to a circular import)

    That is why I have to double the function from p1... here
    You can use talk_to_me() or connect()
    :tablename: str
    :return:
    """

    # talk_to_me()

    connect()

    engine = sqlengine()

    # Todo: how to inherit if_exists to push_df_to_db-function?
    df.to_sql(tablename, con=engine, if_exists='replace', chunksize=100)


def save_sp500_tickers(edit=False):
    """
    With this function we save the current SP500-constituents from time to time and merge it server-side
    with our ratings (manually inserted from SP-webpage).

    The pickle module
    Now, it'd be nice if we could just save this list. We'll use the pickle module for this,
    which serializes Python objects for us. We save its nature within the object.
    :return: tickers, df, lists
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # be cautious: the retrieving of the wiki-table needs too much time - you won't see the dataframe (it will
    # be hidden inside the printed table ;-) - so, only print table in case of urgency...
    # time.sleep(3)
    # print(table.prettify())

    tickers = list()
    securities = []
    urls = []
    sectors = []
    subIndustries = []
    headquarters = []
    dates_added = []
    founded_years = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        security = row.findAll('td')[1].text
        sector = row.findAll('td')[2].text
        subIndustry = row.findAll('td')[3].text
        headquarter = row.findAll('td')[4].text
        dateAdded = row.findAll('td')[5].text
        founded = row.findAll('td')[7].text

        # Todo: debug here: trim away \n

        ticker = ticker.upper().rstrip()
        # print(ticker)  # jeder Ticker hat noch ein trailing \n newline, schauen, ob das noch wichtig wird
        tickers.append(ticker)
        securities.append(security)

        sectors.append(sector)
        # sectors = [row.text for row.findAll('td')[2] in table.findAll('tr')[1:]]     # !!!!! FALSCH Todo!!!!!
        subIndustries.append(subIndustry)
        headquarters.append(headquarter)
        dates_added.append(dateAdded)
        founded_years.append(founded)

    for link in table.findAll('a', {'class': "external text"}):
        url = link.get('href')

        urls.append(url)
        urls2 = [link.get('href') for link in table.findAll('a', {'class': "external text"})]

    # list comprehension...
    #securities = [security for ]
    print(f'Neu gezogene Liste vom {dt.datetime.now()}:\n{tickers}')
    print(f'Neu gezogene Liste vom {dt.datetime.now()}:\n{securities}')
    print(urls)
    print('List comprehen...', urls2)
    print('Comparison urls urls2:', urls == urls2)

    """
    Alternative Lösung:
    
    with open('filename.txt') as f:
        alist = [line.rstrip() for line in f]
    """

    # be cautious with writing...!!!
    if edit is True:
        with open('data/sp500tickers_test.pickle', 'wb') as pickle_out:  # usually u use "as f" for file
            pickle.dump(tickers, pickle_out)
        with open("sp500tickers.pickle", "rb") as pickle_in:
            sp500_list = pickle.load(pickle_in)
    else:
        # wir müssen doch von Zeit zu Zeit die Tickers-Liste erneuern
        # ist alles da ... alphabetisch sortiert...
        with open('data/sp500tickers_2402.pickle', 'wb') as einwecken:  # usually u use "as f" for file
            pickle.dump(tickers, einwecken)

        # Hahaha... es gibt 2 sp500tickers_2402.pickle ... BE AWARE ;-)
        with open('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500tickers_2402.pickle', 'wb') as einwecken:  # usually u use "as f" for file
            pickle.dump(tickers, einwecken)

        with open('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500tickers_2402.pickle', 'rb') as weckglas_open:  # usually u use "as f" for file
            sp500_list_2402 = pickle.load(weckglas_open)

        with open('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500tickers.pickle', "rb") as pickle_in:
            sp500_list = pickle.load(pickle_in)

    print(f'Alte Liste vom Januar 23:\n{sp500_list}')
    print(f'Neue Liste vom Februar 24:\n{sp500_list_2402}')
    '''
    Pickle_in dient nur zum Testen; die tickers list und das pickle object sind gleich - haben beide ein trailing \n
    '''
    """
    # Test-DataFrame for prototyping:
    tickers_prototype = ['MMM', 'AOS', 'ABT']

    securities = pd.Series(data=['3M', 'A. O. Smith', 'Abbott'], index=tickers_prototype, dtype='str')
    sectors = pd.Series(['Industrials', 'Industrials', 'Health Care'], index=tickers_prototype)
    subIndustries = pd.Series(['Industrial Conglomerates', 'Building Products', 'Health Care Equipment'],
                              index=tickers_prototype)
    headquarters = pd.Series(['Saint Paul, Minnesota', 'Milwaukee, Wisconsin', 'North Chicago, Illinois'],
                             index=tickers_prototype)
    dateAdded = pd.Series(['1957-03-04', '2017-07-26', '1957-03-04'],
                          index=tickers_prototype)
    founded = pd.Series([1902, 1916, 1888], index=tickers_prototype)

    data = {
        'Security': securities,
        'Sector': sectors,
        'Sub-Industry': subIndustries,
        'Headquarters': headquarters,
        'Date_added': dateAdded,
        'Founded': founded,
        'exchangeWebsite': None,
        'website': None
    }
    """

    data = {
        'security': securities,
        'url': urls2,
        'sectors': sectors,
        'subindustries': subIndustries,
        'headquarters': headquarters,
        'added': dates_added,
        'founded': founded_years
    }

    # let define Pandas the best dtypes on its own:
    sp500_list_verbose = pd.DataFrame(data=data, index=tickers).convert_dtypes()

    # https://pandas.pydata.org/docs/user_guide/timeseries.html#converting-to-timestamps
    #sp500_list_verbose['Date_added'] = pd.to_datetime(sp500_list_verbose['added'], format='%Y-%m-%d')
    sp500_list_verbose['Date_added'] = pd.to_datetime(sp500_list_verbose['added'], format='mixed')

    #we need a Date-column (timestamp) since our sole function pull_df_from_db tries to parse date:
    sp500_list_verbose['Date'] = dt.datetime.now()

    """
    Zum Testen:
    """
    #push_df_to_db_replace_beautifulsoup(sp500_list_verbose, tablename='sp500_list_verbose_from_wikipedia_test')

    """
        Real replacing:
    """
    #push_df_to_db_replace_beautifulsoup(sp500_list_verbose, tablename='sp500_list_verbose_from_wikipedia_2403')

    #push_df_to_db_replace_beautifulsoup(sp500_list_verbose, tablename='sp500_list_verbose_from_wikipedia')

    return {
        'tickers': tickers,
        'df': sp500_list_verbose,
        'lists': [tickers, securities, sectors, subIndustries, headquarters, dates_added, founded_years]
    }


# CAUTION: don't run - it will mess up your sp500_reload=False argument and relod the list all the time, since it over-
# rides the saved sp500tickers.pickle - file
# save_sp500_tickers()

def sp500_webscraping():
    """
    Extra Funktion nur zum Testen von BeautifulSoup
    :return:
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # be cautious: the retrieving of the wiki-table needs too much time - you won't see the dataframe (it will
    # be hidden inside the printed table ;-) - so, only print table in case of urgency...
    # time.sleep(3)
    print(table.prettify())

    tickers = list()
    securities = []
    urls = []
    sectors = []
    subIndustries = []
    headquarters = []
    dates_added = []
    founded_years = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        security = row.findAll('td')[1].text
        sector = row.findAll('td')[2].text
        subIndustry = row.findAll('td')[3].text
        headquarter = row.findAll('td')[4].text
        dateAdded = row.findAll('td')[5].text
        founded = row.findAll('td')[7].text

        # Todo: debug here: trim away \n

        ticker = ticker.upper().rstrip()
        # print(ticker)  # jeder Ticker hat noch ein trailing \n newline, schauen, ob das noch wichtig wird
        tickers.append(ticker)
        securities.append(security)

        sectors.append(sector)
        # sectors = [row.text for row.findAll('td')[2] in table.findAll('tr')[1:]]     # !!!!! FALSCH Todo!!!!!
        subIndustries.append(subIndustry)
        headquarters.append(headquarter)
        dates_added.append(dateAdded)
        founded_years.append(founded)

    for link in table.findAll('a', {'class': "external text"}):
        url = link.get('href')

        urls.append(url)
        urls2 = [link.get('href') for link in table.findAll('a', {'class': "external text"})]
    print(urls)
    print(urls2)
    print(urls == urls2)

    data = {
        'security': securities,
        'url': urls2,
        'sectors': sectors,
        'subindustries': subIndustries,
        'headquarters': headquarters,
        'added': dates_added,
        'founded': founded_years
    }

    df = pd.DataFrame(data=data, index=tickers)
    print(df)

    # push_df_to_db_replace_beautifulsoup(df, tablename='sp500_list_verbose_from_wikipedia')

"""

hier muss ich nochmal die ungewollten rauslöschen... am 05.03.24 nicht geschafft.....

"""

def pickle_in_tickers_sp500_deprecated_and_new():
    """
    A list of tickers which still contains stock that were removed from sp500 and the current stocks
    for retrieving price data.
    :return:
    """
    real_pickle = 'sp500tickers.pickle'

    with open(real_pickle, "rb") as real_pickle:
        sp500_list_real = pickle.load(real_pickle)
    print(len(sp500_list_real))
    print('')
    print(sp500_list_real)

    """
    every morning, when I download the pricedata, I see tickers which won't download - those tickers feed
    the following set of unwanted tickers:
    """
    unwanted_tickers = {'ATVI', 'ABC', 'BRK.B', 'BF.B', 'DISH', 'RE', 'FRC', 'FISV', 'PKI', 'SIVB',}

    sp500_list_real = [ele for ele in sp500_list_real if ele not in unwanted_tickers]
    print(len(sp500_list_real))
    print('')
    print(sp500_list_real)
    with open('sp500tickers_2403.pickle', 'wb') as pickle_out:  # usually u use "as f" for file
        pickle.dump(sp500_list_real, pickle_out)


def load_mixed_pickles():
    """

    :param saved_pickle:
    :param current_pickle:
    :return:
    """

    saved_pickle = 'data/sp500tickers_saved.pickle'
    current_pickle = 'data/sp500tickers_test.pickle'
    real_pickle = 'C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500tickers_2402.pickle'

    with open(saved_pickle, "rb") as saved_pickle:
        sp500_list_save = pickle.load(saved_pickle)

    with open(current_pickle, "rb") as current_pickle:
        sp500_list_current = pickle.load(current_pickle)

    with open(real_pickle, "rb") as real_pickle:
        sp500_list_real = pickle.load(real_pickle)

    return sp500_list_save, sp500_list_current, sp500_list_real


if __name__ == '__main__':
    #save_sp500_tickers(edit=False)
    #df = save_sp500_tickers()['df']
    #print(df)
    #print(df.dtypes)
    #print('We see one series of the DF-object; sliced out off the return-dict and list of series in the dict:\n',
          #save_sp500_tickers()['df_series'][1], type(save_sp500_tickers()['df_series'][1]))
    #sp500_webscraping()


    sp500_list_save, sp500_list_current, sp500_list_real = load_mixed_pickles()

    #print('Save...\n', sp500_list_save)
    #print('Current...\n', sp500_list_current)
    print(f"""
        Real...
         Das Problem ist, dass wir somit immer die 503 Ticker von Wikipedia haben aber die alten, aus dem Index
         genommenen, verlieren, die ich aber trotzdem weiter als Pricedata ziehen will...
         {len(sp500_list_real)}, {sp500_list_real}
         
    
    """)
    for i in ['VLTO', 'BG', 'KVUE', 'COR', 'EG']:
        if i in sp500_list_real: print(f'{i} is in list')
    #pickle_in_tickers_sp500_deprecated_and_new()
