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

from tqdm import tqdm

yf.pdr_override()
# style.use('ggplot')
plt.style.use('fivethirtyeight')


def do_ml(ticker, use_knn=False):
    """
    Then, we're bringing in the VotingClassifier and RandomForestClassifier.
    The voting classifier is just what it sounds like. Basically, it's a classifier that will let us combine
    many classifiers, and allow them to each get a "vote" on what they think the class of the featuresets is.
    The random forest classifier is just another classifier. We're going to use three classifiers
    in our voting classifier.
    :param use_knn: default False, if func called with param True then we use K-Nearest-Neighbor algorithm, default is
    voting classifier
    :param ticker:
    :return:
    """

    # Todo: wenn ich es richtig verstanden habe, trainiert unser Modell für jeden Tag; es schaut also ob nach einem \n
    # Todo: Tag sich der Preis um 2% verschoben hat, als auch nach 7 Tagen für das 7-Tage Intervall: es könnte also für
    # Todo: Tag 1 eine -1 geben aber für Tag 7 eine 1 als Label, Target, weil die Aktie von Tag 6 bis 7 um 2%
    # Todo gestiegen ist. Wäre es aber nicht besser, immer nur für ein Intervall trainieren zu lassen, also, dass
    # Todo: ich nur für  _d5 (Handelswoche) auf das Target mappen map() lasse? Todo: zweite Funktion schreiben....

    X, y, df, ticker, str_vals, counted_vals = extract_featuresets(ticker)

    # We've got our featuresets and labels, now we want to shuffle them up, train, and then investing:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # metrics = get_scorer_names()
    # print('Possible scorer-metrics: ', metrics)
    # metrics = ['accuracy', 'balanced_accuracy']

    # model (classifier)
    if use_knn is True:
        clf = neighbors.KNeighborsClassifier()
    else:
        clf = VotingClassifier([
            ('lsvc', svm.LinearSVC()),
            ('knn', neighbors.KNeighborsClassifier()),
            ('rfor', RandomForestClassifier())
        ])

    clf.fit(X_train, y_train)

    '''
    cross_validate funzt nicht:
    
    scores = cross_validate(clf, X, y, cv=5, scoring=metrics)

    acc_scores = scores['accuracy']
    bal_scores = scores['balanced_accuracy']

    print("Mean acc of %0.2f with a standard deviation of %0.2f" % (acc_scores.mean(), acc_scores.std()))
    print("Mean bal_acc of %0.2f with a standard deviation of %0.2f" % (bal_scores.mean(), bal_scores.std()))
    '''

    confidence = clf.score(X_test, y_test)

    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    return confidence, predictions   # das hatte ich zuerst vergessen - wenn nichts returned wird, kann man die Funktion zwar
                                    # aufrufen, in einer assigneden Variable wird aber nichts gespeichert


def do_ml_allsp500():
    """

    :return:
    """
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    accuracies = list()
    global main_df
    main_df = pd.DataFrame()

    for count, ticker in enumerate(tqdm(tickers)):

        if count % 10 == 0:
            print(count)
            time.sleep(2)

        try:
            accuracy, predictions = do_ml(ticker, use_knn=False)
        except ValueError as e:
            warnings.warn(
                'Wahrscheinlich nicht min 2 labels vorhanden: {}'.format(e), UserWarning)
            time.sleep(10)
            continue # pass means 'no code to execute here' placeholder for future code, continue just skips the iteration
        accuracies.append(accuracy)

        print("{} accuracy: {}. ".format(ticker, accuracy))
        print('')

        df = pd.DataFrame(columns={'Symbol': ticker, 'Confidence': accuracy})

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    print('Average accuracy: ', mean(accuracies))
    return main_df

    # Todo: build main_df for accuracy and predictions of sp500_ticker
    # Todo: after it push db and pull and then sort dataframe like desc asc min max \n
    # Todo: then to_html and plotten of Top 10


if __name__ == '__main__':
    do_ml('XOM', use_knn=False)
    do_ml('AAPL', use_knn=False)
    do_ml('DE', use_knn=False)
    do_ml('BKNG', use_knn=False)
    do_ml('CVX', use_knn=False)
    do_ml('TDG', use_knn=False)
    main_df = do_ml_allsp500()
    print(main_df)

