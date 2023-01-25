import re

import bs4 as bs
import pickle
import requests


def save_sp500_tickers():
    '''
    The pickle module
    Now, it'd be nice if we could just save this list. We'll use the pickle module for this,
    which serializes Python objects for us. We save its nature within the object.
    :return:
    '''
    headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    #print(table)

    tickers = list()
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        '''
        debug here: trim away \n
        '''
        ticker = ticker.upper().rstrip()
        print(ticker) # jeder Ticker hat noch ein trailing \n newline, schauen, ob das noch wichtig wird
        tickers.append(ticker)
    print(tickers)

    '''
    Alternative Lösung:
    
    with open('filename.txt') as f:
        alist = [line.rstrip() for line in f]
    '''

    with open('sp500tickers.pickle', 'wb') as pickle_out: # usually u use "as f" for file
        pickle.dump(tickers, pickle_out)

    with open("sp500tickers.pickle", "rb") as pickle_in:
        sp500_list = pickle.load(pickle_in)

    print(sp500_list)
    '''
    Pickle_in dient nur zum Testen; die tickers list und das pickle object sind gleich - haben beide ein trailing \n
    '''

    return tickers

# CAUTION: don't run - it will mess up your sp500_reload=False argument and relod the list all the time
# save_sp500_tickers()

