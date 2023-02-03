import os
import datetime as dt
import time
import warnings
from connect import connect
from myutils import sqlengine, sqlengine_pull_from_db, talk_to_me, timestamp, timestamp_onlyday, back_to_the_future
import pickle

import pandas as pd
import yfinance as yf

from tqdm import tqdm

from p5_get_sp500_list import save_sp500_tickers

from p1add_pricedata_to_database import push_df_to_db_replace, push_df_to_db_append, pull_df_from_db

pd.set_option("display.max.columns", None)


def analysts_recommendations(reload_sp500=False):
    """

    :return:
    """
    if reload_sp500 is True:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    tickers = ['DE'] #, 'CVX', 'XOM']
    df_list = []
    for count, ticker in enumerate(tqdm(tickers)):
        # create ticker instance:
        data = yf.Ticker(ticker)
        # show analysts recommendations
        recommendations = data.recommendations
        recommendations_summary = data.recommendations_summary
        # show analysts other work
        analyst_price_target = data.analyst_price_target
        print(f'Recomm.: {recommendations}')
        print(f'Recomm. summary: {recommendations_summary}')
        print(f'Price targets: {analyst_price_target}')


if __name__ == '__main__':
    analysts_recommendations()

