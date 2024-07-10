import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import dash

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table, State
import dash_ag_grid as dag

import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf

import datetime as dt

from investing.myutils import connect, sqlengine, sqlengine_pull_from_db

from investing.p5_get_sp500_list import save_sp500_tickers
from investing.p1add_pricedata_to_database import push_df_to_db_append, pull_df_from_db, push_df_to_db_replace

from dash.dash_table import FormatTemplate
money = FormatTemplate.money(2)


# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': 10,
    'min_rows': None,
    'max_rows': 10,
    'precision': 4,
    'float_format': lambda x: f'{x:.2f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)

currentdatetime = dt.datetime.now()

dash.register_page(__name__, path='/model1', title='Financial Data Model 1',
                   name='10 - Model 1')


def data_collector():
    """

    :return:
    """
    raw_data = pd.read_csv('data_csv_excel/fin_indicators_us_stocks/2018_Financial_Data.csv', index_col='Unnamed: 0')
    print(raw_data.head())
    print(raw_data.describe(include='all'))

    data = raw_data.copy(deep=True)
    print(data.columns)

    data.rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True)
    print(data.columns)
    print(data.axes)


    # GENIAL!
    print(data.dtypes)
    print(data.head())
    print(data.tail())


def preprocessing():
    """
    Cleaning and cleansing: hier mit outliers, nan, categoricals etc. arbeiten.
    :return:
    """


def indicators_model():
    """
    Meine Modellidee - alle Features auf 5 Punkte prüfen...

    Check for OLS-Assumptions...

    y_ttm_hat = earnings + revenue + costs + divausschüttungsquote + debts + free cash flow,dcf, um den risiko-
     freien Zins mit reinzubringen etc..
    :return:
    """


def layout():
    layout_model = dbc.Container([
                     dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.H3(children='Financial Data Model 1',
                                    style={'textAlign': 'center'}),
                            html.Div(children='The idea is... ',
                                     style={'textAlign': 'center'}),
                            html.Br(),
                            html.Hr(),
                            dag.AgGrid(
                                id='table-financial-data',
                                defaultColDef={'resizable': True, 'sortable': True, 'filter': True, 'minWidth': 115},
                                columnSize='sizeToFit',
                                style={"height": "310px"},
                                dashGridOptions={'pagination': True, 'paginationPageSize': 40},
                                className='ag-theme-balham-dark',
                                # alpine is default - alpine-auto-dark ag-theme-quartz ag-theme-balham-dark
                            )
                        ], width=10),
                        dbc.Col([], width=1),
                        ]),
        ])
    return layout_model


if __name__ == '__main__':
    data_collector()
    indicators_model()

