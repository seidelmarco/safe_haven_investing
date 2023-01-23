from urllib.request import Request, urlopen
import json
import ssl
import numpy as np

import pandas as pd
import openpyxl

from tqdm import tqdm
# Bsp. für alle Schleifen/Iterables (list-comprehension):
# sharpe_ratios, wghts = np.column_stack([sharpe_ratios_and_weights(returns) for x in tqdm(range(n))])
import time
from datetime import datetime
from myutils import timestamp

import pickle
from collections import Counter

from p1add_pricedata_to_database import push_df_to_db_replace, push_df_to_db_append, pull_df_from_db

pd.set_option("display.max.columns", None)


def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    # Ignore SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    response = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    dataframe = urlopen(response, context=ctx)
    data = dataframe.read().decode("utf-8")
    return json.loads(data)


def fmp_company_profile():
    """
    Objective: daily updated table with the most important company infos and keymetrics on hand for
    amending the stock-price-predictions
    https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_API_KEY

    Debugging: there is an odd mistake: pd.DataFrame.from_dict reads the json-String and returns a dataframe
    but the type is still string. So we won't be able to use the append-method later on. Better we us some Pandas
    read json methods...
    :return:
    """

    # for testing - just few tickers:
    # best would be to create a table for the selection for that I can populate the table from my Django-admin-panel
    # Caution! No stocks from outside USA! Keep in mind by designing your collection.
    tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY', 'MTNOY', 'BLDP', 'KO', 'DLTR',
               'XOM', 'JNJ', 'KHC', 'MKC', 'MSFT', 'OGN', 'SKT', 'TDG', 'CTRA']
    #tickers = ['DE', 'CTRA']
    print(f'Die Ticker zum Testen sind: {tickers}')
    stopp = input('... hit Enter')

    df_list = []

    for i in tqdm(tickers):
        i.upper()

        api_profile_url = 'https://financialmodelingprep.com/api/v3/profile/' + i + '?apikey=d8285415cbc592b214ce8ffa2b376c23'
        data_profile = get_jsonparsed_data(api_profile_url)
        print(data_profile)
        #stopp = input('... hit Enter')

        # By default the keys of the dict become the DataFrame columns:
        df = pd.DataFrame.from_dict(data_profile)
        df.set_index('symbol', inplace=True)
        df['divyield'] = (df['lastDiv'] / df['price'])*100
        print(df)
        print(type(df)) # <class 'pandas.core.frame.DataFrame'>
        #print('Bis hierin läuft alles; 2 DF - aber jetzt wird nichts zusammengefügt....')
        #stopp = input('... hit Enter')

        df_list.append(df)
    final_df = pd.concat(df_list, axis=0, join='outer')

    return final_df

'''
Example for concat:

data_day_list = []
for i, day in enumerate(list_day):
    data_day = df[df.day == day]
    data_day_list.append(data_day)
final_data_day = pd.concat(data_day_list)

'''

if __name__ == '__main__':
    main_df = fmp_company_profile()
    print(main_df)
    print(type(main_df))
    push_df_to_db_replace(main_df, 'company_keyfacts_selection')
    # Todo: don't work due to parse dates in pull_df....
    # df = pull_df_from_db(sql='company_keyfacts_selection')
    # print(df)
    main_df_sorted = main_df.sort_values(by='divyield', ascending=False)
    print(main_df_sorted.head(30))

