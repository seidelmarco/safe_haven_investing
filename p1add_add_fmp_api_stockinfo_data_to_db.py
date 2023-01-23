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
    :return:
    """
    main_df = pd.DataFrame()


    # for testing - just few tickers:
    # best would be to create a table for the selection for that I can populate the table from my Django-admin-panel
    tickers = ['DE', 'CTRA']
    print(f'Die Ticker zum testen sind: {tickers}')
    stopp = input('... hit Enter')

    df_list = list()
    for i in tqdm(tickers):
        i.upper()

        api_profile_url = 'https://financialmodelingprep.com/api/v3/profile/' + i + '?apikey=d8285415cbc592b214ce8ffa2b376c23'
        data_profile = get_jsonparsed_data(api_profile_url)
        df = pd.DataFrame.from_dict(data_profile)
        print(df)
        df_list = df_list.append(df)

    main_df = pd.concat(df_list, axis=0, join='outer')
    return main_df


fmp_company_profile()
main_df = fmp_company_profile()
print(main_df)

