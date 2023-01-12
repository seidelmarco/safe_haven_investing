'''

p9: preprocessing data for ML
p10 - p11: creating targets for machine learning labels

ML-classifiers: A classifier is the algorithm itself – the rules used by machines to classify data.
It automatically orders or categorizes data into one or more set of "classes". E. g. email-classifier, that scans
emails to filter them by class label: Spam or Not Spam.

We take "featuresets" and try to map them to "labels". We need to convert our existing data to featuresets and labels.

Features: (could be other companies' prices) but here pricing changes for certain days for all companies.
Labels: buy,sell or do nothing regarding a stock - buy if rises 2% in 7 days, sell if falls 2% in 7 days,
in between - do nothing
'''

import bs4 as bs
import pickle
import requests

import time
import datetime as dt
import os  # os is to check for, and create, directories
import warnings

import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

from tqdm import tqdm

# dictionary-subclass for counting many objects at once:
from collections import Counter

import p8_sp500_correlation_table as p8

yf.pdr_override()
# style.use('ggplot')
plt.style.use('fivethirtyeight')


# p8.visualize_data()


def process_data_for_labels(ticker: str):
    '''
    This function will take one parameter: the ticker in question. Each model will be trained on a single company.
    Next, we want to know how many days into the future we need prices for. We're choosing 7 here.
    Now, we'll read in the data for the close prices for all companies that we've saved in the past,
    grab a list of the existing tickers, and we'll fill any missing with 0 for now. This might be something
    you want to change in the future, but we'll go with 0 for now. Now, we want to grab the % change values
    for the next 7 days:

    Todo: currently we read the pricedate from sp500_joined_closes.csv. Instead of updating this file on a daily basis,
    Todo: it would be better to read directly from the db.
    Todo: the sp500_joined_closes.csv is created in p7_combining_sp500data_into_one_dataframe.py
    Todo: but the daily data comes from p6....
    :param ticker:
    :return: tickers, df
    '''
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    # Todo - read sql-table
    # Todo df = pd.read_sql()
    tickers = df.columns.values.tolist()
    # print('Liste aus func "process_data_for_labels": ',tickers)
    df.fillna(0, inplace=True)
    # print(df[['XOM']])

    for i in range(1, hm_days+1): # zwischen Tag 1 und Tag 8
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        '''
        This creates new dataframe columns for our specific ticker in question, using string formatting to create 
        the custom names. The way we're getting future values is with .shift, which basically will shift a column 
        up or down. In this case, we shift a negative amount, which will take that column and, if you could see 
        it visually, it would shift that column UP by i rows. This gives us the future values i days in advanced, 
        which we can calculate percent change against.
        E. g. Day 1: 100, Day 2: 150 = (we lift the column by -i) (150 -100)/100 = 0.5 (50%)
        
        '''

    df.fillna(0, inplace=True)
    return tickers, df


# print(process_data_for_labels('DE'))


def buy_sell_hold(*args):
    '''
    Using args here, so we can take any number of columns here that we want.
    The idea here is that we're going to map this function to a Pandas DataFrame column,
    and that column will be our "label." A -1 is a sell, 0 is hold, and 1 is a buy.
    The *args will be those future price change columns, and we're interested if we see movement
    that exceeds 2% in either direction. Do note, this isn't a totally perfect function.
    For example, price might go up 2%, then fall 2%, and we might not be prepared for that, but it will do for now.
    :param args:
    :return:
    '''
    cols = [c for c in args]
    # set for test purposes on 5 - 10 percent so that you can see unreal exxagerated results - visibility is better
    # default in tutorial: 2 percent
    requirement = 0.05
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


# Exkurs counter:
# Use a string as an argument

# word = input('Enter a long and difficult word: ') # rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz
# counter = Counter(word)
# print(counter)

#df = process_data_for_labels('DE')
#print('Ich sehe alle 503 cols, obwohl ich nur einen Ticker als arg passed habe. ', df)


def extract_featuresets(ticker):
    '''

    :param ticker:
    :return:
    '''

    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))

    # print(df['{}_target'.format(ticker)])
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('')
    print('Results for: ', ticker)
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # The capital X contains our featuresets (daily % changes for every company in the S&P 500).
    X = df_vals.values

    # The lowercase y is our "target" or our "label." Basically what we're trying to map our featuresets to.
    y = df['{}_target'.format(ticker)].values

    return X, y, df


# X, y, df = extract_featuresets('XOM')
# print('X: ', X, 'y: ', y)
# print('')
# print(df)