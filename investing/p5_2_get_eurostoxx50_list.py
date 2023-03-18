# https://en.wikipedia.org/wiki/EURO_STOXX_50

import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import pickle

from myutils import push_df_to_db_replace


def save_eurostoxx50_tickers():
    """
    The pickle module
    Now, it'd be nice if we could just save this list. We'll use the pickle module for this,
    which serializes Python objects for us. We save its nature within the object.
    :return:
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    response = requests.get('https://en.wikipedia.org/wiki/EURO_STOXX_50#Composition', headers=headers)

    soup = BeautifulSoup(response.text, 'lxml')
    # print(soup.prettify())
    table = soup.find_all('table', {'class': 'wikitable sortable'})
    """
        returns eine Liste, die ich in den Indexer jagen kann ;-)
    """
    # print(table)
    table_composition = table[1]
    print(table_composition)

    tickers = list()
    for row in table_composition.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        '''
        debug here: trim away \n
        '''
        ticker = ticker.upper().rstrip()
        tickers.append(ticker)
    # print(tickers)

    '''
    Alternative Lösung:

    with open('filename.txt') as f:
        alist = [line.rstrip() for line in f]
    '''

    with open('eurostoxx50tickers.pickle', 'wb') as pickle_out:  # usually u use "as f" for file
        pickle.dump(tickers, pickle_out)

    with open("eurostoxx50tickers.pickle", "rb") as pickle_in:
        eurostoxx50_list = pickle.load(pickle_in)

    print(eurostoxx50_list)
    print(len(eurostoxx50_list))

    '''
    Pickle_in dient nur zum Testen; die tickers list und das pickle object sind gleich - haben beide ein trailing \n
    '''

    joined_frames_list = []

    for count, row in enumerate(tqdm(table_composition.findAll('tr')[1:])):
        ticker = row.findAll('td')[0].text
        name = row.findAll('td')[2].text
        industry = row.findAll('td')[5].text
        ticker = ticker.upper().rstrip()

        data = {
            'Symbol': ticker,
            'Name': name,
            'Industry': industry,
        }

        df = pd.DataFrame(data, columns=['Symbol', 'Name', 'Industry'], index=[ticker])
        df.set_index('Symbol', inplace=True)
        joined_frames_list.append(df)

    main_df = pd.concat(joined_frames_list, axis=0, join='outer')
    #print(main_df)

    return tickers, main_df

# CAUTION: don't run after the initial time - it will mess up your sp500_reload=False argument
# and reload the list all the time


if __name__ == '__main__':
    save_eurostoxx50_tickers()
    tickers, df = save_eurostoxx50_tickers()
    push_df_to_db_replace(df, 'eurostoxx50_tickers_names_industries')
    print(df)
