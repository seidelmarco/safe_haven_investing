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

from p91011_preprocessing_data_for_ml import extract_featuresets
from p12_ml_against_sp500_prices import do_ml, do_ml_allsp500

from tqdm import tqdm

yf.pdr_override()
# style.use('ggplot')
plt.style.use('fivethirtyeight')

pd.set_option("display.max.columns", None)


#with open('sp500tickers.pickle', 'rb') as f:
    #tickers = pickle.load(f)
'''
frames = []
tickers = ['DE', 'AAPL', 'XOM', 'CVX']

for count, ticker in enumerate(tqdm(tickers)):

    if count % 10 == 0:
        print(count)
        time.sleep(2)

    try:
'''

ticker = ['DE']
ticker_2 = ['AAPL']
ticker_3 = ['XOM']
ticker_4 = ['CVX']


        #for ticker in tqdm(tickers):
# hist data comes from extract_featuresets
X, y, df, ticker, str_vals, counted_vals = extract_featuresets(ticker)
print(ticker, counted_vals)
df_hist = pd.DataFrame.from_dict(counted_vals, orient='index').T
df_hist.rename(columns={'1': 'hist_buy', '-1': 'hist_sell', '0': 'hist_hold'}, inplace=True)
df_hist['Symbol'] = ticker
df_hist.set_index('Symbol', inplace=True)
print(df_hist)

# the predicted data (labels) comes from do_ml
accuracy, predictions = do_ml(ticker, use_knn=False)
predictions = Counter(predictions)
buy_predicted = predictions[1]  # key 1
sell_predicted = predictions[-1]
do_nothing = predictions[0]
print(f'buy: {buy_predicted}, sell: {sell_predicted}, do nothing: {do_nothing}')

data = {
    'Symbol': ticker,
    'Confidence': accuracy,
    'Buy_predicted': buy_predicted,
    'Sell_predicted': sell_predicted,
    'Hold_predicted': do_nothing
}

df_1 = pd.DataFrame(data, columns=['Symbol', 'Confidence', 'Buy_predicted', 'Sell_predicted', 'Hold_predicted'])

df_1.set_index(['Symbol'], inplace=True)

print(df_1)

frames = [df_hist, df_1]
joined_frame = pd.concat(frames, axis=1, join='outer')
print(joined_frame)

'''

hier geht frame 2 los:


'''

# hist data comes from extract_featuresets
X, y, df, ticker, str_vals, counted_vals = extract_featuresets(ticker_2)
print(ticker, counted_vals)
df_hist2 = pd.DataFrame.from_dict(counted_vals, orient='index').T
df_hist2.rename(columns={'1': 'hist_buy', '-1': 'hist_sell', '0': 'hist_hold'}, inplace=True)
df_hist2['Symbol'] = ticker
df_hist2.set_index('Symbol', inplace=True)
print(df_hist2)

# the predicted data (labels) comes from do_ml
accuracy, predictions = do_ml(ticker_2, use_knn=False)
predictions = Counter(predictions)
buy_predicted = predictions[1]  # key 1
sell_predicted = predictions[-1]
do_nothing = predictions[0]
print(f'buy: {buy_predicted}, sell: {sell_predicted}, do nothing: {do_nothing}')

data_2 = {
    'Symbol': ticker_2,
    'Confidence': accuracy,
    'Buy_predicted': buy_predicted,
    'Sell_predicted': sell_predicted,
    'Hold_predicted': do_nothing
}

df_2 = pd.DataFrame(data_2, columns=['Symbol', 'Confidence', 'Buy_predicted', 'Sell_predicted', 'Hold_predicted'])
df_2.set_index('Symbol', inplace=True)

print(df_2)

frames = [df_hist2, df_2]

joined_frame2 = pd.concat(frames, axis=1, join='outer')
print(joined_frame2)

'''

hier geht frame 3 los:


'''

# hist data comes from extract_featuresets
X, y, df, ticker, str_vals, counted_vals = extract_featuresets(ticker_3)
print(ticker, counted_vals)
df_hist3 = pd.DataFrame.from_dict(counted_vals, orient='index').T
df_hist3.rename(columns={'1': 'hist_buy', '-1': 'hist_sell', '0': 'hist_hold'}, inplace=True)
df_hist3['Symbol'] = ticker
df_hist3.set_index('Symbol', inplace=True)
print(df_hist3)

# the predicted data (labels) comes from do_ml
accuracy, predictions = do_ml(ticker_3, use_knn=False)
predictions = Counter(predictions)
buy_predicted = predictions[1]  # key 1
sell_predicted = predictions[-1]
do_nothing = predictions[0]
print(f'buy: {buy_predicted}, sell: {sell_predicted}, do nothing: {do_nothing}')

data_3 = {
    'Symbol': ticker_3,
    'Confidence': accuracy,
    'Buy_predicted': buy_predicted,
    'Sell_predicted': sell_predicted,
    'Hold_predicted': do_nothing
}

df_3 = pd.DataFrame(data_3, columns=['Symbol', 'Confidence', 'Buy_predicted', 'Sell_predicted', 'Hold_predicted'])
df_3.set_index('Symbol', inplace=True)

print(df_3)

frames = [df_hist3, df_3]

joined_frame3 = pd.concat(frames, axis=1, join='outer')
print(joined_frame3)

'''

hier geht frame 4 los:


'''

# hist data comes from extract_featuresets
X, y, df, ticker, str_vals, counted_vals = extract_featuresets(ticker_4)
print(ticker, counted_vals)
df_hist4 = pd.DataFrame.from_dict(counted_vals, orient='index').T
df_hist4.rename(columns={'1': 'hist_buy', '-1': 'hist_sell', '0': 'hist_hold'}, inplace=True)
df_hist4['Symbol'] = ticker
df_hist4.set_index('Symbol', inplace=True)
print(df_hist4)

# the predicted data (labels) comes from do_ml
accuracy, predictions = do_ml(ticker_4, use_knn=False)
predictions = Counter(predictions)
buy_predicted = predictions[1]  # key 1
sell_predicted = predictions[-1]
do_nothing = predictions[0]
print(f'buy: {buy_predicted}, sell: {sell_predicted}, do nothing: {do_nothing}')

data_4 = {
    'Symbol': ticker_4,
    'Confidence': accuracy,
    'Buy_predicted': buy_predicted,
    'Sell_predicted': sell_predicted,
    'Hold_predicted': do_nothing
}

df_4 = pd.DataFrame(data_4, columns=['Symbol', 'Confidence', 'Buy_predicted', 'Sell_predicted', 'Hold_predicted'])
df_4.set_index('Symbol', inplace=True)

print(df_4)

frames = [df_hist4, df_4]

joined_frame4 = pd.concat(frames, axis=1, join='outer')
print(joined_frame4)

joined_frames = [joined_frame, joined_frame2, joined_frame3, joined_frame4]

main_df = pd.concat(joined_frames, axis=0, join='outer')
print('')
print("""
*************************
Das sollte mein Endergebnis sein:
***********************++
""")
print(main_df.sort_values(by='Buy_predicted', ascending=False))

'''
    except ValueError as e:
        warnings.warn(
            'Wahrscheinlich nicht min 2 labels vorhanden: {}'.format(e), UserWarning)
        time.sleep(10)
        continue # pass means 'no code to execute here' placeholder for future code, continue just skips the iteration

    if main_df.empty:
        main_df = df
    else:
        # join raises value error: column overlap - no suffix provided
        main_df = main_df.join(df, how='outer')
        # main_df = main_df.merge(df, on='Symbol', how='outer')
        # main_df = main_df.concat(axis=1)
        #main_df = main_df.append(df)

print(main_df)
'''

#class BuildADataframe(unittest.TestCase):
 #   def test_something(self):
  #      self.assertEqual(True, False)  # add assertion here


#if __name__ == '__main__':
    #unittest.main()
    #BuildADataframe()
