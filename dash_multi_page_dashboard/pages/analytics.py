import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table

import plotly.express as px

import pandas as pd

from investing.p1add_pricedata_to_database import pull_df_from_db

from dash_multi_page_dashboard.preps_and_tests.unittest_returns_table import normalized_returns_from_database

import datetime as dt

from dash.dash_table import FormatTemplate

import colorlover

# the FormatTemplate provides the following predefined templates:
money = FormatTemplate.money(2)
percentage = FormatTemplate.percentage(2)

# for the daily returns (third table) I use another template since we need more digits after the floating point:
money_returns = FormatTemplate.money(5, sign='')
percentage_returns = FormatTemplate.percentage(4)

# auskommentieren, wenn ich zum Testen dieses Skript solo nutzen möchte...
dash.register_page(__name__,
                   path='/analytics',
                   title='Stock Analytics',
                   name='Stock Analytics')

"""
Colourpicker:
"""

"""
https://encycolorpedia.de/708090
"""

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
Collect data...
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

    df = pull_df_from_db(sql='sp500_adjclose')
    df = df.reset_index()
    df.copy()

    df_vol = pull_df_from_db(sql='sp500_volume')
    df_vol = df_vol.reset_index()
    df_vol.copy()

    filtered_data = df

    # assign new columns to a DataFrame
    filtered_data_newdateformat = filtered_data.assign(Date=lambda filtered_data: pd.to_datetime(filtered_data['Date'], format='%Y-%m-%d'))
    filtered_data_newdateformat.sort_values(by='Date', inplace=True)

    # am besten schon hier sortieren, weil es sonst cris-cross Linien in plotly gibt (plotly sortiert nicht)
    # Todo: hier muss ich noch sortieren, um die jumbled (durcheinandergewürfelten) lines zu vermeiden
    #  - hat funktioniert...
    df_vol = df_vol.sort_values(by='Date')

    #print(type(filtered_data_newdateformat['Date']))

    filtered_data_loc_one_ticker_all_rows = df.loc[:, ticker]

    filtered_data_only_ticker_col = filtered_data_newdateformat[ticker]

    return df, filtered_data, filtered_data_newdateformat, filtered_data_loc_one_ticker_all_rows, filtered_data_only_ticker_col, df_vol


""" #### Layout #### """
filtered_data = filtered_data_from_df()[2]
filtered_data.columns[1:]
df = filtered_data_from_df()[2]
df_table = df.sort_values(by='Date', ascending=False)

# This is a simple way to extract the date - NICHT date() verwenden! TypeError: Series object is not callable:
df_table['Date'] = df_table['Date'].dt.date

df_table['id'] = df_table.index

choices = df_table[['id', 'Date', 'NEM', 'MSFT', 'AAPL', 'TDG', 'CARR', 'CVX', 'XOM', 'CTRA', 'COST', 'DE', 'DLTR', 'IT',
                    'JNJ', 'LUMN', 'MCD','MRNA', 'NVDA', 'PCAR', 'PWR', 'TRGP']]

# formating the table values - DataTable.columns is a list of dicts with keys:

# columns = [{'id': c, 'name': c} for c in df.columns]

columns = [
    dict(id='Date', name='Date'),
    dict(id='NEM', name='NEM', type='numeric', format=money),
    dict(id='MSFT', name='MSFT', type='numeric', format=money),
    dict(id='AAPL', name='AAPL', type='numeric', format=money),
    dict(id='TDG', name='TDG', type='numeric', format=money),
    dict(id='CARR', name='CARR', type='numeric', format=money),
    dict(id='CVX', name='CVX', type='numeric', format=money),
    dict(id='XOM', name='XOM', type='numeric', format=money),
    dict(id='CTRA', name='CTRA', type='numeric', format=money),
    dict(id='COST', name='COST', type='numeric', format=money),
    dict(id='DE', name='DE', type='numeric', format=money),
    dict(id='DLTR', name='DLTR', type='numeric', format=money),
    dict(id='IT', name='IT', type='numeric', format=money),
    dict(id='JNJ', name='JNJ', type='numeric', format=money),
    dict(id='LUMN', name='LUMN', type='numeric', format=money),
    dict(id='MCD', name='MCD', type='numeric', format=money),
    dict(id='MRNA', name='MRNA', type='numeric', format=money),
    dict(id='NVDA', name='NVDA', type='numeric', format=money),
    dict(id='PCAR', name='PCAR', type='numeric', format=money),
    dict(id='PWR', name='PWR', type='numeric', format=money),
    dict(id='TRGP', name='TRGP', type='numeric', format=money),

]


columns_returns = [
    dict(id='Date', name='Date'),
    dict(id='NEM', name='NEM', type='numeric', format=money_returns),
    dict(id='MSFT', name='MSFT', type='numeric', format=money_returns),
    dict(id='AAPL', name='AAPL', type='numeric', format=money_returns),
    dict(id='TDG', name='TDG', type='numeric', format=money_returns),
    dict(id='CARR', name='CARR', type='numeric', format=money_returns),
    dict(id='CVX', name='CVX', type='numeric', format=money_returns),
    dict(id='XOM', name='XOM', type='numeric', format=money_returns),
    dict(id='CTRA', name='CTRA', type='numeric', format=money_returns),
    dict(id='COST', name='COST', type='numeric', format=money_returns),
    dict(id='DE', name='DE', type='numeric', format=money_returns),
    dict(id='DLTR', name='DLTR', type='numeric', format=money_returns),
    dict(id='IT', name='IT', type='numeric', format=money_returns),
    dict(id='JNJ', name='JNJ', type='numeric', format=money_returns),
    dict(id='LUMN', name='LUMN', type='numeric', format=money_returns),
    dict(id='MCD', name='MCD', type='numeric', format=money_returns),
    dict(id='MRNA', name='MRNA', type='numeric', format=money_returns),
    dict(id='NVDA', name='NVDA', type='numeric', format=money_returns),
    dict(id='PCAR', name='PCAR', type='numeric', format=money_returns),
    dict(id='PWR', name='PWR', type='numeric', format=money_returns),
    dict(id='TRGP', name='TRGP', type='numeric', format=money_returns),

]

df_vol = filtered_data_from_df()[5]

df_returns_norm = normalized_returns_from_database()

"""


Wrapped-up reusable functions for conditional formatting in Dash Tables - use what you need...


"""


def highlight_max_row(df_max_per_row):
    """
    it is better for some cases of conditional formatting to wrap-up the following list in an outsourced function which
    stores all your formatting and returns a list
    :param df_max_per_row: pass your df - in this case: choices
    :param df:
    :return:
    """
    # be aware of the select_dtypes('number')-function, 'Date won't work...
    df_numeric_columns = df_max_per_row.select_dtypes('number').drop(['id'], axis=1)

    return [
            {
               'if': {
                        'filter_query': '{{id}} = {}'.format(i),
                        'column_id': col
                     },
                     'backgroundColor': 'transparent',  ##3D9970
                     'color': 'tomato',
                     'fontWeight': 'bold'
            }
            # idxmax(axis=1) finds the max indices of each row
            for (i, col) in enumerate(df_numeric_columns.idxmax(axis=1))
            ]


def highlight_top2_values_every_row(df_top2_values, nlargest=2):
    """
    !!! Does not make any sense in my table with absolute stock prices since you only see the highest grown absolute
    prices like TDG and COST - this function only make sense with normalized returns...

    pandas.df.select_dtypes ('number') is a good way to jsut filter the numeric cols omitting the date-col
    :param nlargest:
    :param df_top2_values:
    :return:
    """
    numeric_columns = df_top2_values.select_dtypes('number').drop(['id'], axis=1).columns
    styles = []
    for i in range(len(df_top2_values)):
        row = df_top2_values.loc[i, numeric_columns].sort_values(ascending=False)
        for j in range(nlargest):
            styles.append({
                'if': {
                    'filter_query': '{{id}} = {}'.format(i),
                    'column_id': row.keys()[j]
                },
                'backgroundColor': '#39CCCC',  #transparent0
                'color': 'white',
                #'fontWeight': 'bold'
            })
    return styles


def highlight_max_val_in_table(df_max_val):
    """
    Highlighting the Maximum Value in the Table

    A new alternative to catch the 'id'-col prior to choosing the columns:

    Todo: This function is usefully for the highest return - rewrite the original prices-Table into a returns-Table...
    :param df_max_val:
    :return:
    """
    if 'id' in df_max_val:
        numeric_columns = df_max_val.select_dtypes('number').drop(['id'], axis=1)
    else:
        numeric_columns = df_max_val.select_dtypes('number')
    max_across_numeric_columns = numeric_columns.max()
    max_across_table = max_across_numeric_columns.max()
    styles = []
    for col in max_across_numeric_columns.keys():
        if max_across_numeric_columns[col] == max_across_table:
            styles.append({
                'if': {
                    'filter_query': "{{{col}}} = {value}".format(col=col, value=max_across_table),
                    'column_id': col
                },
                'backgroundColor': '#39CCCC',   ##B10DC9 - lighter than rebeccaPurple | 39CCCC - kind of light cyan
                'color': 'white'
            })
    return styles


def highlight_range_of_values(df_range):
    """
    Let's break down \{{\{col}}}. We want the final expression to look something like {2017} > 5 & {2017} < 10
    where 2017 is the name of the column. Since we're using .format(), we need to escape the brackets,
    so {2017} would be {{2017}}. Then, we need to replace 2017 with {col} for the find-and-replace,
    so becomes\{{\{col}}}.format(col=col)

    :param df_range:
    :return: best use-case in return-tables :-)
    """
    # IMMER diese beiden Zeilen, um Datum und 'id' rauszuwerfen - sonst gibt es einen Callback-Fehler:
    if 'id' in df_range:
        numeric_columns = df_range.select_dtypes('number').drop(['id'], axis=1)
    else:
        numeric_columns = df_range.select_dtypes('number')
    # styles = [].append() == see below...
    styles = [{
        'if': {
            'filter_query': '{{{col}}} > 100 && {{{col}}} <= 300'.format(col=col),
            'column_id': col
        },
        'backgroundColor': '#B10DC9',  ##B10DC9 - lighter than rebeccaPurple | 39CCCC - kind of light cyan
        'color': 'white'
    } for col in numeric_columns.columns]

    return styles


def highlight_top_bottom_10_percent_by_col(df_percent):
    """
    Highlighting Top 10% or Bottom 10% of Values by Column

    For bottom use quantile(.1)

    :param df_percent:
    :return:
    """
    if 'id' in df_percent:
        num_cols = df_percent.select_dtypes('number').drop(['id'], axis=1)
    else:
        num_cols = df_percent.select_dtypes('number')
    # Todo: AttributeError: 'Series' object has no attribute 'iteritems' - deprecated: use items()
    styles = [{
        'if': {
            'filter_query': '{{{}}} >= {}'.format(col, value),
            'column_id': col
        },
        'backgroundColor': 'goldenRod',
        'color': 'white'
    } for (col, value) in num_cols.quantile(.9).items()]
    return styles


def highlight_above_below_average(df_average):
    """
    Highlighting Values above Average and Below Average - Here, the highlighting is done per column.
    :param df_average:
    :return:
    """
    if 'id' in df_average:
        num_cols = df_average.select_dtypes('number').drop(['id'], axis=1)
    else:
        num_cols = df_average.select_dtypes('number')
    styles = [{
        'if': {
            'filter_query': '{{{}}} <= {}'.format(col, value),
            'column_id': col
        },
        'backgroundColor': 'paleTurquoise',
        'color': 'white'
    } for (col, value) in num_cols.quantile(.5).items()]
    return styles


def highlight_above_below_average_per_table(df_average_table):
    """
    Here done with the average of the whole table...

    DAS ergibt wirklich nur in einem normalized table Sinn.

    :param df_average_table:
    :return:
    """
    if 'id' in df_average_table:
        num_cols = df_average_table.select_dtypes('number').drop(['id'], axis=1)
    else:
        num_cols = df_average_table.select_dtypes('number')
    # funktioniert nur mit dieser Doppelkonstruktion - wir brauchen den Average von allen columns:
    df_mean = num_cols.mean().mean()
    styles = ([{
        'if': {
            'filter_query': '{{{}}} > {}'.format(col, df_mean),
            'column_id': col
        },
        'backgroundColor': '#3D9970',
        'color': 'white'
    } for col in num_cols.columns]
    +
    [{
        'if': {
            'filter_query': '{{{}}} <= {}'.format(col, df_mean),
            'column_id': col
        },
        'backgroundColor': '#FF4136',
        'color': 'white'
    } for col in num_cols.columns])
    return styles


def discrete_background_color_bins(df_heatmap, n_bins=8, cols='all'):
    """

    Highlighting Cells by Value with a Colorscale Like a Heatmap

    This recipe shades cells with style_data_conditional and creates a legend with HTML components.
    You'll need to pip install colorlover to get the colorscales.

    :param df_heatmap:
    :param n_bins:
    :param cols: we can pass a list of chosen columns as argument
    :return:
    """

    # list comprehension for the bounds/limits - you need n+1 bounds since your first bound start at zero
    # you iterate with i over the range of n_bins + 1 and then in a list comprehension (short for [].append()
    # you divide the whole (1.0) by the n_bins and in every iteration you multiply by i (what stops at n_bins + 1)
    # the result with 4 bins is  a list -> Bounds: [0.0, 0.25, 0.5, 0.75, 1.0]
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    print('Bounds:', bounds)

    print('DF-raw-data:\n', df_heatmap)

    # default from function 'all' columns
    if cols == 'all':
        if 'id' in df_heatmap:
            # my databases only stores stock-prices-tables with Date and Id - so we have to drop those data-types:
            df_numeric_columns = df_heatmap.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df_heatmap.select_dtypes('number')
    else:
        # we can pass as argument into the function a list of chosen columns
        df_numeric_columns = df_heatmap[cols]

    print('Num-cols or chosen columns:\n', df_numeric_columns)

    # double-decker: we identify the max-min-value of every col and then the max-min-value of the whole collection/table
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()

    print('************************************')
    print(f'df_max: {df_max}, df_min: {df_min}')
    print('************************************')

    # again list comprehension
    # result of ranges considering 9 bound (0 - 8 i):
    # [-10.0, -7.125, -4.25, -1.375, 1.5, 4.375, 7.25, 10.125, 13.0]
    # we translated the bounds into ranges/bins by multiplying with the bounds,
    # the bounds acted like % (12,5%, 50%, 87,5% etc.)
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    print('Ranges:\n', ranges)

    # our return values - the well known lists for the conditional_data in Dash:
    styles = []
    legend = []

    # range 1 - 9 for 8 bins:
    for i in range(1, len(bounds)):
        # go back to index 0 by (i-1)
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        # colorLover takes the bins and is indexing starting with 0 to 8:
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        # very CLEVER! Since the first 50% of the colorscales are too pale, the font-color just becomes white
        # when the scales are getting darker - for the contrast :-)
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                # the backgroundColor comes at runtime always as a new one, since the styles are a nested loop in the
                # main loop
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))

# All CSS-Bootstrap-grid-system classes can be assigned to the html.Div([]) elements, within their className property.
"""
*****************************************
https://dash.plotly.com/datatable/style
*****************************************

These are the priorities of style_* props, in decreasing order:

    style_data_conditional
    style_data
    style_filter_conditional
    style_filter
    style_header_conditional
    style_header
    style_cell_conditional
    style_cell

"""

# ...at this point we start with dash: you can either use the variable layout or a function with the variable layout...

# that's the only recipe what you have to call prior to the layout since it returns a tuple of two return-list
# below-mentioned there is only one return value allowed - we need the styles in the layout
(styles, legend) = discrete_background_color_bins(choices, n_bins=8, cols='all')    # 'all' | ['MSFT', 'AAPL', 'IT', 'NVDA']
(styles_returns, legend_returns) = discrete_background_color_bins(df_returns_norm, n_bins=8, cols='all')    # 'all' | ['MSFT', 'AAPL', 'IT', 'NVDA']

layout = dbc.Container([
    dbc.Row([
        html.H3(children='Stock Analysis: Price- and Returntables:', style={'textAlign': 'center'}),
        html.Br(),
        dbc.Col([], width=1),
        dbc.Col([
                dash_table.DataTable(data=choices.to_dict(orient='records'),
                                     # format is a dict with keys:
                                     # try this later: columns=[{'name': i, 'id': i} for i in df.columns if i != 'id'],
                                     columns=columns,        # [{'name': i, 'id': i} for i in choices], # das klappt nicht wegen der Formatierung
                                     selected_columns=['CTRA'], page_size=10,
                                     sort_action='native',  # custom | native | none
                                     editable=False,
                                     style_table={'width': '100%', 'overflowX': 'auto'},
                                     style_header={
                                        'backgroundColor': colors['slate-grey-darker'],
                                        #'fontFamily': 'Arial',
                                        'font-size': '1rem',
                                        'fontWeight': 'bold',
                                        'color': 'orange',
                                        'border': '0px transparent',
                                        'textAlign': 'right'},

                                     #css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                     style_cell={
                                        'height': 'auto',
                                        # all three widths are needed
                                        'minWidth': '100px', 'width': '120px', 'maxWidth': '150px',
                                        'whiteSpace': 'normal',
                                         # oder...sieht nicht gut aus bei meinen Aktienpreisen - wird zu eng...

                                         #style_cell={
                                         #'width': '{}%'.format(len(choices.columns)),
                                         #'textOverflow': 'ellipsis',
                                         #'overflow': 'hidden',
                                         # }
                                        'backgroundColor': 'transparent',  #colors['medium-blue-grey'],
                                        #'fontFamily': 'Arial',
                                        'font-size': '0.9rem',
                                        'color': colors['invertiert'],
                                        'border': '1px transparent',
                                        'textAlign': 'right'},
                                     style_header_conditional=[{
                                        'if': {'column_id': 'Date'},
                                        'textAlign': 'left', 'padding-left': '1%',
                                     } for c in choices.columns],
                                     style_cell_conditional=[{
                                        'if': {'column_id': 'Date'},
                                        'textAlign': 'left', 'padding-left': '1%',
                                     },
                                         {
                                          'if': {'column_id': choices.columns[-1]},
                                          'padding-right': '1%',
                                         }],
                                     # for striped rows use style_data and ...data_conditional:
                                     style_data={
                                         'color': colors['invertiert'],
                                         'backgroundColor': 'transparent',
                                         'border': '1px solid #a3998e',
                                     },

                                     # hier kommt der Mega-hack: weil style_data_conditional eine Liste ist,
                                     # können wir eine Funktion schreiben, die den DF auswertet und die Ergebnisse
                                     # in eine leere [] appended:
                                     style_data_conditional=[
                                         {
                                             'if': {'row_index': 'odd'},
                                             'backgroundColor': 'transparent',
                                         },
                                         {
                                             'if': {'column_id': 'Date'},
                                             'backgroundColor': colors['slate-grey']
                                         },
                                         {
                                             'if': {
                                                 'filter_query': '{CTRA} > 27 && {CTRA} < 28',
                                                 'column_id': 'CTRA'
                                             },
                                             'backgroundColor': 'tomato',
                                             'color': 'white'
                                         },
                                         {
                                             'if': {
                                                 'column_id': 'TDG',
                                                 # since using .format, escape { with {{
                                                 'filter_query': '{{TDG}} = {}'.format(df_table['TDG'].max())
                                             },
                                             'backgroundColor': '#85144b',
                                             'color': 'white'
                                         },
                                         {
                                             'if': {
                                                 'row_index': 5,    # number | odd | even
                                                 'column_id': 'AAPL'
                                             },
                                             'backgroundColor': 'hotpink',
                                             'color': 'white'
                                         },
                                         {
                                             'if': {
                                                'filter_query': '{{id}} = {}'.format(choices['id'].max()),     # matching rows of a hidden col with the id 'id'
                                                'column_id': 'CARR'
                                             },
                                             'backgroundColor': 'RebeccaPurple',
                                             'color': 'white'
                                         },
                                         {
                                             'if': {
                                                 # hier können wir auch mit returns oder mavg arbeiten...
                                                 'filter_query': '{MSFT} > {AAPL}', # comparing columns to each other
                                                 'column_id': 'MSFT'
                                             },
                                             'backgroundColor': 'dodgerBlue',     # #3D9970
                                             'color': 'white'
                                         },
                                         # Todo: upcoming dicts - Formatting und Highlighting recipes - PLACEHOLDER
                                         {
                                             'if': {
                                                 # highlighting a complete row with a min-value
                                                 # idea: highlight all stockprices, when the Gold-price was low:
                                                 # since we have only the sp500 in this prototype, we take NEM
                                                 # Newmont Mining Corp (the largest gold miner) as placeholder
                                                 'filter_query': '{{NEM}} = {}'.format(choices['NEM'].min())
                                             },
                                             'backgroundColor': '#FF4136',     # #3D9970
                                             'color': 'white'
                                         },
                                     ]
                                     +
                                     # The hack is to add multiple lists...
                                     [
                                        {
                                             'if': {
                                                 # Highlighting the Top Five or Bottom Three Values in a Column
                                                 'filter_query': '{{TRGP}} = {}'.format(i),
                                                 'column_id': 'TRGP'
                                             },
                                             'backgroundColor': 'cornflowerBlue',     # #3D9970
                                             'color': 'white'
                                        }
                                        for i in choices['TRGP'].nlargest(5)
                                     ]
                                     +
                                     [
                                        {
                                             'if': {
                                                 # Highlighting the Top Five or Bottom Five Values in a Column
                                                 'filter_query': '{{TRGP}} = {}'.format(i),
                                                 'column_id': 'TRGP'
                                             },
                                             'backgroundColor': 'lime',     # #3D9970
                                             'color': 'white'
                                        }
                                        for i in choices['TRGP'].nsmallest(5)
                                     ]

                                     ,
                                     cell_selectable=True,
                                     column_selectable="multi",
                                     style_as_list_view=False,  # True: you only see horizontal lines
                                     ),
            html.Br(),
            html.Hr(),
                ], width=10),
        dbc.Col([], width=1),
    ]),
    # Todo: *****************************************************
    #
    # SECOND TABLE:
    #
    # Todo: *****************************************************
    dbc.Row([
        html.H3(children='...', style={'textAlign': 'center'}),
        html.Br(),
        dbc.Col([], width=1),
        dbc.Col([
                html.Pre(children='paleTurquoise are values below the average, bold red fonts are the max values...',
                         style={'float': 'right'}),
                # for the object "legend" you find the whole html-code above-mentioned in the heatmap-function
                html.Div(legend, style={'float': 'right'}),
                dash_table.DataTable(data=choices.to_dict(orient='records'),
                                     # format is a dict with keys:
                                     columns=columns,        # [{'name': i, 'id': i} for i in choices], # das klappt nicht wegen der Formatierung
                                     selected_columns=['CTRA'], page_size=10,
                                     sort_action='native',  # custom | native | none
                                     editable=False,
                                     style_table={'width': '100%', 'overflowX': 'auto'},
                                     style_header={
                                        'backgroundColor': colors['slate-grey-darker'],
                                        #'fontFamily': 'Arial',
                                        'font-size': '1rem',
                                        'fontWeight': 'bold',
                                        'color': 'orange',
                                        'border': '0px transparent',
                                        'textAlign': 'right'},

                                     #css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                     style_cell={
                                        'height': 'auto',
                                        # all three widths are needed
                                        'minWidth': '100px', 'width': '120px', 'maxWidth': '150px',
                                        'whiteSpace': 'normal',
                                         # oder...sieht nicht gut aus bei meinen Aktienpreisen - wird zu eng...

                                         #style_cell={
                                         #'width': '{}%'.format(len(choices.columns)),
                                         #'textOverflow': 'ellipsis',
                                         #'overflow': 'hidden',
                                         # }
                                        'backgroundColor': 'transparent',  #colors['medium-blue-grey'],
                                        #'fontFamily': 'Arial',
                                        'font-size': '0.9rem',
                                        'color': colors['invertiert'],
                                        'border': '1px transparent',
                                        'textAlign': 'right'},
                                     style_header_conditional=[{
                                        'if': {'column_id': 'Date'},
                                        'textAlign': 'left', 'padding-left': '1%',
                                     } for c in choices.columns],
                                     style_cell_conditional=[{
                                        'if': {'column_id': 'Date'},
                                        'textAlign': 'left', 'padding-left': '1%',
                                     },
                                         {
                                          'if': {'column_id': choices.columns[-1]},
                                          'padding-right': '1%',
                                         }],
                                     # for striped rows use style_data and ...data_conditional:
                                     style_data={
                                         'color': colors['invertiert'],
                                         'backgroundColor': 'transparent',
                                         'border': '1px solid #a3998e',
                                     },

                                     # hier kommt der Mega-hack: weil style_data_conditional eine Liste ist,
                                     # können wir eine Funktion schreiben, die den DF auswertet und die Ergebnisse
                                     # in eine leere [] appended:
                                     style_data_conditional=[
                                         {
                                             'if': {'row_index': 'odd'},
                                             'backgroundColor': 'transparent',
                                         },
                                         {
                                             'if': {'column_id': 'Date'},
                                             'backgroundColor': colors['slate-grey']
                                         }
                                     ]
                                     +
                                     # The hack is to add multiple lists...
                                     [
                                        {
                                             'if': {
                                                 # Highlighting the Top Five or Bottom Three Values in a Column
                                                 'filter_query': '{{TRGP}} = {}'.format(i),
                                                 'column_id': 'TRGP'
                                             },
                                             'backgroundColor': 'cornflowerBlue',     # #3D9970
                                             'color': 'white'
                                        }
                                        for i in choices['TRGP'].nlargest(5)
                                     ]
                                     +
                                     [
                                        {
                                             'if': {
                                                 # Highlighting the Top Five or Bottom Five Values in a Column
                                                 'filter_query': '{{TRGP}} = {}'.format(i),
                                                 'column_id': 'TRGP'
                                             },
                                             'backgroundColor': 'lime',     # #3D9970
                                             'color': 'white'
                                        }
                                        for i in choices['TRGP'].nsmallest(5)
                                     ]
                                     +
                                     highlight_max_row(choices)
                                     +
                                     highlight_max_val_in_table(choices)
                                     #+
                                     #highlight_top_bottom_10_percent_by_col(choices)
                                     #+
                                     #highlight_above_below_average(choices)
                                     #+
                                     # highlight_above_below_average_per_table(choices)
                                     +
                                     styles
                                     ,
                                     cell_selectable=True,
                                     column_selectable="multi",
                                     style_as_list_view=False,  # True: you only see horizontal lines
                                     ),
            html.Br(),
            html.Hr(),
                ], width=10),
        dbc.Col([], width=1),
    ]),
    # Todo: *****************************************************
    #
    # THIRD TABLE:
    #
    # Todo: *****************************************************
    dbc.Row([
        html.H3(children='...', style={'textAlign': 'center'}),
        html.Br(),
        dbc.Col([], width=1),
        dbc.Col([
                html.Pre(children='Testing conditional formatting like percentages in the scope of a returns table...',
                         style={'float': 'right'}),
                # for the object "legend" you find the whole html-code above-mentioned in the heatmap-function
                html.Div(legend_returns, style={'float': 'right'}),
                dash_table.DataTable(data=df_returns_norm.to_dict(orient='records'),
                                     # format is a dict with keys:
                                     columns=columns_returns,        # [{'name': i, 'id': i} for i in choices], # das klappt nicht wegen der Formatierung
                                     #columns=[{'name': i, 'id': i} for i in df_returns_norm], # das klappt nicht wegen der Formatierung
                                     selected_columns=['CTRA'], page_size=10,
                                     sort_action='native',  # custom | native | none
                                     editable=False,
                                     style_table={'width': '100%', 'overflowX': 'auto'},
                                     style_header={
                                        'backgroundColor': colors['slate-grey-darker'],
                                        #'fontFamily': 'Arial',
                                        'font-size': '1rem',
                                        'fontWeight': 'bold',
                                        'color': 'orange',
                                        'border': '0px transparent',
                                        'textAlign': 'right'},

                                     #css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                     style_cell={
                                        'height': 'auto',
                                        # all three widths are needed
                                        'minWidth': '100px', 'width': '120px', 'maxWidth': '150px',
                                        'whiteSpace': 'normal',
                                         # oder...sieht nicht gut aus bei meinen Aktienpreisen - wird zu eng...

                                         #style_cell={
                                         #'width': '{}%'.format(len(choices.columns)),
                                         #'textOverflow': 'ellipsis',
                                         #'overflow': 'hidden',
                                         # }
                                        'backgroundColor': 'transparent',  #colors['medium-blue-grey'],
                                        #'fontFamily': 'Arial',
                                        'font-size': '0.9rem',
                                        'color': colors['invertiert'],
                                        'border': '1px transparent',
                                        'textAlign': 'right'},
                                     style_header_conditional=[{
                                        'if': {'column_id': 'Date'},
                                        'textAlign': 'left', 'padding-left': '1%',
                                     } for c in df_returns_norm.columns],
                                     style_cell_conditional=[{
                                        'if': {'column_id': 'Date'},
                                        'textAlign': 'left', 'padding-left': '1%',
                                     },
                                         {
                                          'if': {'column_id': df_returns_norm.columns[-1]},
                                          'padding-right': '1%',
                                         }],
                                     # for striped rows use style_data and ...data_conditional:
                                     style_data={
                                         'color': colors['invertiert'],
                                         'backgroundColor': 'transparent',
                                         'border': '1px solid #a3998e',
                                     },

                                     # hier kommt der Mega-hack: weil style_data_conditional eine Liste ist,
                                     # können wir eine Funktion schreiben, die den DF auswertet und die Ergebnisse
                                     # in eine leere [] appended:
                                     style_data_conditional=[
                                         {
                                             'if': {'row_index': 'odd'},
                                             'backgroundColor': 'transparent',
                                         },
                                         {
                                             'if': {'column_id': 'Date'},
                                             'backgroundColor': colors['slate-grey']
                                         }
                                     ]
                                     +
                                     # The hack is to add multiple lists...
                                     [
                                        {
                                             'if': {
                                                 # Highlighting the Top Five or Bottom Three Values in a Column
                                                 'filter_query': '{{TRGP}} = {}'.format(i),
                                                 'column_id': 'TRGP'
                                             },
                                             'backgroundColor': 'cornflowerBlue',     # #3D9970
                                             'color': 'white'
                                        }
                                        for i in df_returns_norm['TRGP'].nlargest(5)
                                     ]
                                     +
                                     [
                                        {
                                             'if': {
                                                 # Highlighting the Top Five or Bottom Five Values in a Column
                                                 'filter_query': '{{TRGP}} = {}'.format(i),
                                                 'column_id': 'TRGP'
                                             },
                                             'backgroundColor': 'lime',     # #3D9970
                                             'color': 'white'
                                        }
                                        for i in df_returns_norm['TRGP'].nsmallest(5)
                                     ]
                                     #+
                                     #highlight_max_row(df_returns_norm)
                                     #+
                                     #highlight_max_val_in_table(df_returns_norm)
                                     #+
                                     #highlight_top_bottom_10_percent_by_col(df_returns_norm)
                                     #+
                                     #highlight_above_below_average(df_returns_norm)
                                     #+
                                     # highlight_above_below_average_per_table(df_returns_norm)
                                     +
                                     styles_returns
                                     ,
                                     cell_selectable=True,
                                     column_selectable="multi",
                                     style_as_list_view=False,  # True: you only see horizontal lines
                                     ),
            html.Br(),
            html.Hr(),
                ], width=10),
        dbc.Col([], width=1),
    ]),



    dbc.Row([
        html.H3(children='Stock Analysis: price-, volume-charts and moving averages', style={'textAlign': 'center'}),
        html.Br(),
        dbc.Col([html.Div(children='')], width=2),
        dbc.Col([html.Div(children=''),
        html.Div([
            html.Div([]),
            html.Div([html.P(children='Price-Chart with RangeSlider und Selectors:')]),

            dcc.Dropdown(options=df.columns[1:], value='CTRA', id='chosen_ticker', clearable=True,
                         placeholder='You see the index...', style={'background-color': colors['back-white']}),
            html.Br(),
            dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=df['Date'].min().date(),
                    max_date_allowed=df['Date'].max().date(),
                    start_date=df['Date'].min().date(),
                    end_date=df['Date'].max().date(),
                    start_date_placeholder_text=df['Date'].min().date(),
                    end_date_placeholder_text=df['Date'].max().date(),
                ),
            # Using the plotly.express library, we build the histogram chart and assign it to the figure property
            # of the dcc.Graph. This displays the histogram in our app. We also can build up the figure property
            # from scratch.

            # the next line is for static plotting - for test purposes:
            # dcc.Graph(id='time-series-chart', figure=px.line(df, x='Date', y='CTRA'))

            # better we leave the figure property an empty {} and shift the plotting to the callback:
            dcc.Graph(id='time-series-chart', figure={'data': [], 'layout': {}, 'frames': []}),
            html.Br(),
            html.Hr(),
            dcc.Graph(id='volume-chart', figure={'data': [], 'layout': {}, 'frames': []})
        ])
    ], width=8),
        dbc.Col([], width=2)
    ])
])


# Add controls to build the interaction
@callback(
    Output(component_id='time-series-chart', component_property='figure'),
    Output(component_id='volume-chart', component_property='figure'),
    Input(component_id='chosen_ticker', component_property='value'),
    Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')
)
def update_time_series_chart(col_chosen, start_date, end_date):
    df_filtered_dates = df.query('Date >= @start_date and Date <= @end_date')
    dfvol_filtered_dates = df_vol.query('Date >= @start_date and Date <= @end_date')

    price_chart_figure = px.line(df_filtered_dates, x='Date', y=col_chosen)
    price_chart_figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        )
    )

    volume_chart_figure = px.bar(dfvol_filtered_dates, x='Date', y=col_chosen)
    volume_chart_figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        )
    )

    return price_chart_figure, volume_chart_figure


if __name__ == '__main__':
    df = filtered_data_from_df()[2]
    print(df)
    only_one_column = filtered_data_from_df()[4]
    print(only_one_column)