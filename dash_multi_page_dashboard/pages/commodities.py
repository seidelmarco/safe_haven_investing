import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table

from investing.p1add_pricedata_to_database import pull_df_from_db

import pandas as pd

import datetime as dt

from dash.dash_table import FormatTemplate

import colorlover

# the FormatTemplate provides the following predefined templates:
money = FormatTemplate.money(2)
percentage = FormatTemplate.percentage(2)

# for the daily returns (third table) I use another template since we need more digits after the floating point:
money_returns = FormatTemplate.money(5)
percentage_returns = FormatTemplate.percentage(4)

dash.register_page(__name__, path='/commodities', title='Commodities Analysis', name='Commodities Analysis', order=3)

# All CSS-Bootstrap-grid-system classes can be assigned to the html.Div([]) elements, within their className property.

colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'white': 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)',
    'slate-grey': '#5c6671',
    'slate-grey-invertiert': '#a3998e',
    'slate-grey-darker': '#4b5259',
    'dusk-white': '#CFCFCF',
    'back-white': '#F7F7F7',
    'anthrazitgrau': '#373f43',
    'invertiert': '#c8c0bc'

}

"""
Data:
"""


def filtered_data_from_df(start_date=dt.datetime.now()-dt.timedelta(15), end_date=dt.datetime.now(), ticker: str = 'CTRA'):
    """
    Basisdaten, um Dataframes zu konstruieren.
    #Todo: ich muss mit Queries noch start- und enddate filtern
    :param start_date: dient zum Testen in diesem Soloskript
    :param end_date: dient zum Testen in diesem Soloskript
    :param ticker: dient zum Testen in diesem Soloskript
    :return:
    """

    settings = {
        'max_columns': None,
        'min_rows': None,
        'max_rows': 20,
        'precision': 8,
        'float_format': lambda x: f'{x:.4f}'
    }

    for option, value in settings.items():
        pd.set_option(f'display.{option}', value)

    df = pull_df_from_db(sql='commodities_ohlc')
    df = df.reset_index()
    df.copy()

    filtered_data = df

    # assign new columns to a DataFrame
    filtered_data_newdateformat = filtered_data.assign(Date=lambda filtered_data: pd.to_datetime(filtered_data['Date'], format='%Y-%m-%d'))
    filtered_data_newdateformat.sort_values(by='Date', ascending=False, inplace=True)

    # am besten schon hier sortieren, weil es sonst cris-cross Linien in plotly gibt (plotly sortiert nicht)
    # Todo: hier muss ich noch sortieren, um die jumbled (durcheinandergewürfelten) lines zu vermeiden
    #  - hat funktioniert...
    #df_vol = df_vol.sort_values(by='Date')

    #print(type(filtered_data_newdateformat['Date']))

    return {'df': df,
            'filtered_data': filtered_data,
            'filtered_newdate': filtered_data_newdateformat,
            }


df = filtered_data_from_df()['filtered_newdate']
df['Date'] = df['Date'].dt.date

df_choices = df[['Date', 'GC=F_Adj_Close', 'GC=F_daily_pct_chng', 'PL=F_Adj_Close', 'PL=F_daily_pct_chng',
                 'HG=F_Adj_Close', 'HG=F_daily_pct_chng', 'CL=F_Adj_Close', 'CL=F_daily_pct_chng']]


columns_date = [dict(id='Date', name='Date')]


columns = [
    dict(id='GC=F_Adj_Close', name='Gold', type='numeric', format=money),
    dict(id='GC=F_daily_pct_chng', name='Gold %', type='numeric', format=percentage_returns),
    dict(id='PL=F_Adj_Close', name='Platinum', type='numeric', format=money),
    dict(id='PL=F_daily_pct_chng', name='Platinum %', type='numeric', format=percentage_returns),
    dict(id='HG=F_Adj_Close', name='Copper', type='numeric', format=money),
    dict(id='HG=F_daily_pct_chng', name='Copper %', type='numeric', format=percentage_returns),
    dict(id='CL=F_Adj_Close', name='Crude Oil', type='numeric', format=money),
    dict(id='CL=F_daily_pct_chng', name='Crude Oil %', type='numeric', format=percentage_returns),
]


def layout():
    layout_commod = dbc.Container([
        dbc.Row([
            html.Div([
                html.H1(children='This is our Commodities page', className="text-center fs-3", style={'color': colors['invertiert']}),
                html.Div("""Keeping track of commodities. The math: df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
                        df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']""", style={'textAlign': 'center'}),
                html.Br()
            ])
        ]),
        dbc.Row([
            dbc.Col([html.Div(children=' ')], width=2),
            dbc.Col([html.Div(children=' '),
                dash_table.DataTable(
                    columns=(columns_date + columns),
                    data=df_choices.to_dict(orient='records'),
                    page_size=12,
                    sort_action='native',
                    filter_action='native',
                    style_table={'overflowX': 'auto'}),
                     ], width=8),
            dbc.Col([html.Div(children=' ')], width=2),
        ]),
    ], fluid=True)

    return layout_commod


if __name__ == '__main__':
    df = filtered_data_from_df()['filtered_newdate']
    print(df)


