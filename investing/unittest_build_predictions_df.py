import unittest
import time

import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as pdr

import warnings

from sklearn import svm, neighbors
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import get_scorer_names
from statistics import mean

import yfinance as yf

import pickle
import requests
from collections import Counter
from investing.myutils import sum

from p1add_pricedata_to_database import push_df_to_db_replace, push_df_to_db_append, pull_df_from_db
from p91011_preprocessing_data_for_ml import extract_featuresets
from p12_ml_against_sp500_prices import do_ml, do_ml_allsp500

from tqdm import tqdm

yf.pdr_override()
# style.use('ggplot')
plt.style.use('fivethirtyeight')

pd.set_option("display.max.columns", None)


with open('sp500tickers.pickle', 'rb') as f:
    tickers = pickle.load(f)
    print(tickers)
    # you need to split the list of ticker-strings into a list of lists, otherwise the loop won't work
    # use a list comprehension
    tickers_list = [str(val).split(",") for val in tickers]
    print(tickers_list)

joined_frames_list = []
#stopp = input('...')

#tickers_list = [['DE'], ['AAPL'], ['XOM'], ['CVX']]
# tickers_list = [['XOM'], ['CVX'], ['BRK.B']]

for count, ticker in enumerate(tqdm(tickers_list)):

    if count % 10 == 0:
        print(count)
        time.sleep(2)

    # hist data comes from extract_featuresets
    X, y, df, ticker_ex, str_vals, counted_vals = extract_featuresets(ticker)
    df_hist = pd.DataFrame.from_dict(counted_vals, orient='index').T
    df_hist.rename(columns={'1': 'hist_buy', '-1': 'hist_sell', '0': 'hist_hold'}, inplace=True)
    df_hist['Symbol'] = ticker
    print(df_hist)
    df_hist.set_index('Symbol', inplace=True)

    # the predicted data (labels) comes from do_ml
    try:
        accuracy, predictions = do_ml(ticker, use_knn=False)
    except ValueError as e:
        warnings.warn(
            'Wahrscheinlich nicht min 2 labels vorhanden: {}'.format(e), UserWarning)
        time.sleep(10)
        continue
    predictions = Counter(predictions)
    buy_predicted = predictions[1]  # key 1
    sell_predicted = predictions[-1]
    do_nothing = predictions[0]

    data = {
        'Symbol': ticker,
        'Confidence': accuracy,
        'Buy_predicted': buy_predicted,
        'Sell_predicted': sell_predicted,
        'Hold_predicted': do_nothing
    }

    df_pred = pd.DataFrame(data, columns=['Symbol', 'Confidence', 'Buy_predicted', 'Sell_predicted',
                                          'Hold_predicted'])

    df_pred.set_index('Symbol', inplace=True)

    frames = [df_hist, df_pred]
    joined_frame = pd.concat(frames, axis=1, join='outer')
    joined_frame['Summe hist'] = joined_frame['hist_buy'] + joined_frame['hist_sell'] + joined_frame['hist_hold']
    # print(joined_frame)
    joined_frames_list.append(joined_frame)

main_df = pd.concat(joined_frames_list, axis=0, join='outer')

print('')
print("""
*************************
Das sollte mein Endergebnis sein:

aus irgendeinem Grund löscht es die DF wie BRK.B raus - das ist momentan gut aber könnte später zu Problemem führen!
***********************++
""")
main_df_sorted = main_df.sort_values(by='Buy_predicted', ascending=False)
print(main_df_sorted.head(30))

push_df_to_db_replace(main_df_sorted, 'sp500_predicted_buys')
'''
Example for concat:

data_day_list = []
for i, day in enumerate(list_day):
    data_day = df[df.day == day]
    data_day_list.append(data_day)
final_data_day = pd.concat(data_day_list)

'''
