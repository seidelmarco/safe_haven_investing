import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np

from investing.p1add_pricedata_to_database import pull_df_from_db

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

import unittest

import datetime as dt


def filtered_data_from_df(start_date='2023-10-01', end_date='2023-10-27', ticker: str = 'CTRA'):

    df = pull_df_from_db(sql='sp500_adjclose')
    df.reset_index(inplace=True)

    filtered_data = df.copy()

    # assign new columns to a DataFrame
    filtered_data_newdateformat = filtered_data.assign(Date=lambda filtered_data: pd.to_datetime(filtered_data['Date'], format='%Y-%m-%d'))
    filtered_data_newdateformat.sort_values(by='Date', inplace=True)

    print(filtered_data_newdateformat)
    print(type(filtered_data_newdateformat['Date']))


    filtered_data_loc_one_ticker_all_rows = df.loc[:, ticker]

    filtered_data_only_ticker_row = df[ticker]

    return df, filtered_data, filtered_data_newdateformat, filtered_data_loc_one_ticker_all_rows, filtered_data_only_ticker_row

"""
class DFTests(unittest.TestCase):
    #class for running unittests - Achtung, wenn der Test läuft, läuft der Server nicht! Auslagern.

    def setUP(self):
        # Your setUp
        TEST_INPUT_DIR = '/'
        test_file_name = ''
        try:
            data = pd.read_csv(TEST_INPUT_DIR + test_file_name, sep=',', header=0)
        except IOError:
            print('cannot open file..')
        # ich vermute, fixture ist eine Methode, die wir on the fly erstellt haben...
        self.fixture = data

    def testFiltered_data_from_df(self):
        #Test that the dataframe read in equals what you expect
        foo = df
        assert_frame_equal(self.fixture, foo)
"""

df, filtered_data, filtered_data_newdateformat, filtered_data_loc_one_ticker_all_rows, filtered_data_only_ticker_row = filtered_data_from_df()

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Stock price analysis', style={'textAlign':'center'}),
    html.P('Select stock:'),
    #dcc.Dropdown(options=['GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'MSFT'], value='MSFT', id='ticker', clearable=False),
    dcc.Dropdown(options=df.columns[1:], value='MSFT', id='ticker', clearable=False),
    dcc.DatePickerRange(
        id='date-range',
        min_date_allowed=df['Date'].min().date(),
        max_date_allowed=df['Date'].max().date(),
        start_date=df['Date'].min().date(),
        end_date=df['Date'].max().date(),
        start_date_placeholder_text=df['Date'].min().date(),
        end_date_placeholder_text=df['Date'].max().date(),
    ),
    dcc.Graph(id='time-series-chart')
])

@callback(
    Output('time-series-chart', 'figure'),
    Input('ticker', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_graph(ticker, start_date, end_date):
    """
    Die Variante mit px.data.stocks() funktioniert zum Prototyping nur mit den
    options=['GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'MSFT']
    :param ticker:
    :return:
    """
    #filtered_dates = df.query('Date >= @start_date and Date <= @end_date')
    start_date = start_date
    end_date = end_date
    #df_px = px.data.stocks()  # replace with your own data source
    #fig = px.line(df_px, x='date', y=ticker)
    ##data = df.loc[start_date:end_date, :]

    # Filter based on the date filters
    #df_filtered_dates = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), :].copy()    #Bug: macht komische Streifen
    # oder alternativ mit query
    """
    Tipp: der wichtigste Punkt war, VOR dem plotten der Linie einen neuen DF zu filtern, sodass Daten und 
    Ticker-values die gleiche Länge haben.
    """
    df_filtered_dates = filtered_data_newdateformat.query('Date >= @start_date and Date <= @end_date')
    #print(df_filtered_dates[ticker])


    #fig = px.line(df, x='Date', y=ticker)
    fig = px.line(df_filtered_dates, x='Date', y=ticker, title='Time series with rangeslider and selectors')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector = dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        )
    )
    return fig


if __name__ == '__main__':
    app.run(port='8053', debug=True)
    #DFTests()
    df, filtered_data, filtered_data_newdateformat, filtered_data_loc_one_ticker_all_rows, filtered_data_only_ticker_row = filtered_data_from_df()
    #print('Rawdata:\n', df)
    #print('Filtered:\n', filtered_data)
    #print('filtered_data_newdateformat\n', filtered_data_newdateformat)
    #print('filtered_data_loc_one_ticker_all_rows\n', filtered_data_loc_one_ticker_all_rows)
    #print('filtered_data_only_ticker_row\n', filtered_data_only_ticker_row)


    # Test the queries:
    start_date = '2023-10-01'
    end_date = dt.datetime.now()
    ticker = 'AAPL'
    filtered_dates = filtered_data_newdateformat.query('Date >= @start_date and Date <= @end_date')

    print(filtered_dates)

    # Filter based on the date filters
    df_filtered_dates = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), :].copy()
    print("""

        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


        """)
    print(df_filtered_dates)
    print(df_filtered_dates[ticker])


