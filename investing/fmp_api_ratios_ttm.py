from urllib.request import Request, urlopen
import json
import ssl
import numpy as np
import os

import pandas as pd
import openpyxl

from tqdm import tqdm
# Bsp. für alle Schleifen/Iterables (list-comprehension):
# sharpe_ratios, wghts = np.column_stack([sharpe_ratios_and_weights(returns) for x in tqdm(range(n))])
import time
from datetime import datetime
from myutils import timestamp, tickers_list

import pickle
from collections import Counter
import hidden

from p5_get_sp500_list import save_sp500_tickers

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


def fmp_ratiosttm(reload_sp500 = False):
    """

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

    # for testing - just few tickers:
    # best would be to create a table for the selection for that I can populate the table from my Django-admin-panel
    # Caution! No stocks from outside USA! Keep in mind by designing your collection.
    tickers = tickers_list()
    #tickers = ['DE', 'CTRA']
    print(f'Die Ticker zum Testen sind: {tickers}')
    stopp = input('... hit Enter')

    key = hidden.secrets_fmp()
    df_list = []

    for ticker in tqdm(tickers):
        ticker.upper()

        ratiosttm_url = 'https://financialmodelingprep.com/api/v3/ratios-ttm/' + ticker + '?apikey=' + key

        data_ratiosttm = get_jsonparsed_data(ratiosttm_url)
        print(data_ratiosttm)
        # stopp = input('... hit Enter')

        # By default, the keys of the dict become the DataFrame columns:
        df = pd.DataFrame.from_dict(data_ratiosttm)
        df['symbol'] = ticker
        df.set_index('symbol', inplace=True)
        print(df)
        print(type(df))  # <class 'pandas.core.frame.DataFrame'>

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
    main_df = fmp_ratiosttm()
    print(main_df)
    print(type(main_df))
    push_df_to_db_replace(main_df, 'fmp_ratios_ttm')
    # Todo: doesn't work due to parse dates in pull_df....
    # df = pull_df_from_db(sql='fmp_ratios_ttm')