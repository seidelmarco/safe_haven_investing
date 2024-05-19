"""
Before you dive into writing tests, you’ll want to first make a couple of decisions:

        What do you want to test?
        Are you writing a unit test or an integration test?

        Then the structure of a test should loosely follow this workflow:

        Create your inputs
        Execute the code being tested, capturing the output
        Compare the output with an expected result

        For this application, you’re testing sum(). There are many behaviors in sum() you could check, such as:

        Can it sum a list of whole numbers (integers)?
        Can it sum a tuple or set?
        Can it sum a list of floats?
        What happens when you provide it with a bad value, such as a single integer or a string?
        What happens when one of the values is negative?
"""

import os
import inspect
import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np

from investing.connect import connect
from investing.myutils import sqlengine, sqlengine_pull_from_db, talk_to_me, timestamp, timestamp_onlyday, back_to_the_future

from investing.p1add_pricedata_to_database import pull_df_from_db, push_df_to_db_append, push_df_to_db_replace

import datetime as dt

from investing.myutils import timestamp

import unittest

from openpyxl import writer
import openpyxl

# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': 10,
    'min_rows': None,
    'max_rows': 20,
    'precision': 6,
    'float_format': lambda x: f'{x:.4f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)


def pull_df_from_db_wo_dates(sql='sp500_adjclose'):
    """
    # Todo : Funktion umschreiben, dass ich aus allen Tabellen und ausgewählte Spalten beim callen wählen kann
    :sql: default is table 'sp500_adjclose' - you can pass any table from the db as argument
    :return:
    """
    connect()
    engine = sqlengine_pull_from_db()

    # an extra integer-index-column is added
    # df = pd.read_sql(sql, con=engine)
    # Column 'Date' is used as index
    df = pd.read_sql(sql, con=engine)

    return df


def sp500_constituents_allocation_optimization():
    """

    :return:
    """
    # 1. We retrieve a current sp500-constituents-list from wikipedia - since it is retrieved with
    # Beautiful Soup and pushed to the database, it takes quite a while (todo later to find a quicker solution)

    # New solution - scraping already done in another script:
    df_sp500_verbose = pull_df_from_db(sql='sp500_list_verbose_from_wikipedia')

    # meine Standardlösung "pull_df_from_db" braucht 'Date' und setzt es als Index
    # wir müssen vor dem Merge alles resetten und 'Date' droppen
    df_sp500_verbose.reset_index(inplace=True)
    df_sp500_verbose.drop(['Date'], axis=1, inplace=True)

    # rename the col with the name 'index'
    df_sp500_verbose.set_index('index', inplace=True)
    df_sp500_verbose.index.name = 'Symbol'

    print('Dataframe sp500_verbose:\n', df_sp500_verbose)

    # 2. Either we use a pickle as tickers-list or a sliced df-series for downloading pricedata:
    #with open('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/data/sp500tickers_test.pickle', 'rb') as current_pickle:
        #sp500_list_current = pickle.load(current_pickle)

    tickers = df_sp500_verbose.index.tolist()

    #print('Sliced tickers from Dataframe:\n', type(tickers), len(tickers), tickers)

    """
    We have now this first result:
    1.  an up-to-date-df pulled from DB with all current SP500 constituents; we have to push it onto the DB
    from time to time by using the function from p5_get_sp500_list
    2. then we have a ready-made list of all current SP500-tickers sliced from the df
    """

    # 3. We have this old xls-spreadsheet. Goal is to retrieve the ratings and to concat it with the
    # current sp500-constituents. NaNs will be filled later manually directly in the dataframe-xls or via a dash-table
    df_ratings = pd.read_excel('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500_dfs/sources/sp500ratings.xlsx')
    df_ratings.set_index('Symbol', inplace=True)
    print(df_ratings)

    # 4. Construct DF from two lists/dataframes:
    # https://realpython.com/pandas-merge-join-and-concat/
    outer_merged_maindf = pd.merge(df_sp500_verbose, df_ratings, on='Symbol', how='outer')

    # BEWARE: for portfolio optimization we need the left merge (only with the current constituents)
    left_merged_maindf = pd.merge(df_sp500_verbose, df_ratings, on='Symbol', how='left')
    print('##############################')
    print('Outer merge:')
    print('\n', outer_merged_maindf)
    print('##############################')
    print('Left merge:')
    print('\n', left_merged_maindf)
    print('##############################')

    # Be aware: for merging the DFs into a main_df we need the index, but afterwards we have ro reset_index
    # thus we can use also the Symbol-Column as data-argument in our Dash-Table:
    outer_merged_maindf.reset_index(inplace=True)
    left_merged_maindf.reset_index(inplace=True)
    push_df_to_db_replace(left_merged_maindf, 'sp500_leftmerged_old_ratings')

    outer_merged_maindf.to_excel('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500_dfs/main_df_sp500_constituents_ratings.xlsx', engine='openpyxl')
    left_merged_maindf.to_excel('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500_dfs/leftmerged_df_sp500_constituents_ratings.xlsx', engine='openpyxl')

    return outer_merged_maindf, left_merged_maindf


def merge_sp500_constituents_w_new_ratings():
    """
    We aim at establishing a df "available stocks" merged from the wikipedia list plus manually added ratings
    from the sp500 website
    :return:
    """
    # 1. Retrieve the two tables from db - kommt aus Dash-Editable Table

    df_new_ratings = pull_df_from_db(sql='sp500_new_ratings', dates_as_index=False)
    print(df_new_ratings)

    #Todo: merge on symbol and rating.... (concat, join, merge) - for production we need the LEFT merge:
    #left_df = sp500_constituents_allocation_optimization()[0]
    #left_df = sp500_constituents_allocation_optimization()[1]
    #left_df = left_df.reset_index().rename(columns={'Symbol': 'symbol', 'Rating': 'rating'})

    #wenn wir den oberen Schritt schon im Dash-Programm getätigt haben, brauchen wir hier nur noch von der DB pullen:
    left_df = pull_df_from_db_wo_dates('sp500_leftmerged_old_ratings')
    left_df = left_df.reset_index().rename(columns={'Symbol': 'Symbol', 'Rating': 'LT-Rating'})

    left_df = left_df.drop('level_0', axis=1)

    print(f"""

        SP500_leftmerged_old_ratings:

        {left_df}""")

    deadend = input('Press Enter for showing the steps: ')

    main_df = pd.merge(left_df,
                       # inner just for checking-purposes:
                       df_new_ratings, on=(['Symbol']), how='outer',
                       left_on=None, right_on=None, left_index=False, right_index=False,
                       sort=False, suffixes=("_orig", "_dupl"), copy=None, indicator=None) # symbol, rating, left outer....

    # every DF we upload on DB needs a 'Date'-col (due to the parsing-rule in pull_from_db)
    main_df['Date'] = dt.datetime.now()

    print(main_df)

    deadend2 = input('Press Enter for showing the steps: ')


    #Todo: push to db, auf db UPDATE WHERE rating_orig IS NULL...
    #replace or append????
    push_df_to_db_replace(main_df, 'sp500_constituents_updated_ratings')
    #push_df_to_db_append(main_df, 'sp500_constituents_updated_ratings')

    print(main_df)
    # KeyError: 'symbol' - vermutl in df_new_ratings - KeyError: 'rating' - Groß- Kleinschreibung beachten!!!!

    # mein BX scheint zweimal da zu sein - nochmal neu mergen...
    # at[row,col]
    # print('BX' in main_df['symbol'].values)

    result_scalar = main_df.loc[main_df['Symbol'] == 'BALL']
    print(result_scalar)

    # To select rows whose column value is in an iterable, some_values, use isin: value must be in a list e.g.!
    # result_iterable = main_df.loc[main_df['symbol'].isin(['bx'])]
    # print(result_iterable)

    # Todo: https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values

    # favorite for advanced filtering :-)
    # mit df.query() - Achtung! Case sensitive
    #result_scalars = main_df.query('symbol == "BX" | rating_x == "AAA" | rating_y == "AAA"')
    #print(result_scalars)




    main_df.to_excel('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500_dfs/sp500_constituents_merged_ratings.xlsx', engine='openpyxl')


    #Todo: das alles nun in dash-portfolio_optimization eintragen......

def available_stocks():
    """
    retrieve from DB all ratings: cleansing and cleaning - preprocessing...
    :return:
    """
    # retrieve final table and prepare it for final dataframe:
    df_available_stocks_raw = pull_df_from_db(sql='sp500_constituents_updated_ratings_merged')
    #print(df_available_stocks_raw)

    df_avail_cols = df_available_stocks_raw[['Symbol', 'security', 'Name', 'Sector', 'LT-Rating_orig']]
    #print(df_avail_cols.index.to_flat_index())

    # checking for missing values - no missing values here; very good:
    print(df_avail_cols.info())

    # drop missing values (mv):
    print(df_avail_cols.describe())
    print(df_avail_cols.isnull().sum())
    data_no_mv = df_avail_cols.dropna(subset=['LT-Rating_orig'])
    print(data_no_mv.isnull().sum())

    print(df_avail_cols)
    print(data_no_mv)
    print(data_no_mv.shape)
    print(data_no_mv.axes)


if __name__ == '__main__':
    #df = sp500_constituents_allocation_optimization()[0]
    #print(df)

    # Either or - one after another of these both:
    # merge_sp500_constituents_w_new_ratings()
    available_stocks()

