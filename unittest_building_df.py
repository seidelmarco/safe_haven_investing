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

global main_df
main_df = pd.DataFrame()
ticker = ['DE']
data = [0.4567]

X, y, df_featureset = extract_featuresets(ticker)
print(f'X: {X}, y: {y}')
print('Dataframe Featureset: ')
print(df_featureset)

accuracy = do_ml(ticker, use_knn=False)

df = pd.DataFrame(columns={'Symbol': ticker,
                           'Confidence': accuracy,
                           'Buy': 'default',
                           'Sell': 'default',
                           'Hold': 'default',
                           'Buy_predicted': 'default',
                           'Sell_predicted': 'default',
                           'Hold_predicted': 'default'
                           })
df['Symbol'] = ticker
df['Confidence'] = accuracy
df.set_index('Symbol', inplace=True)
print(df)

if main_df.empty:
    main_df = df
else:
    main_df = main_df.join(df, how='outer')

print(main_df)


#class BuildADataframe(unittest.TestCase):
 #   def test_something(self):
  #      self.assertEqual(True, False)  # add assertion here


#if __name__ == '__main__':
    #unittest.main()
    #BuildADataframe()
