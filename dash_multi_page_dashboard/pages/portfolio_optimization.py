import dash

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table, State
import dash_ag_grid as dag

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf

import datetime as dt

from investing.myutils import connect, sqlengine, sqlengine_pull_from_db

from investing.p5_get_sp500_list import save_sp500_tickers
from investing.p1add_pricedata_to_database import push_df_to_db_append, pull_df_from_db, push_df_to_db_replace

import dash_multi_page_dashboard.preps_and_tests.unittest_portfolio_balancing_montecarlosim as montecarlo

from dash_multi_page_dashboard.preps_and_tests.unittest_portfolio_balancing_montecarlosim import lineplot_daily_returns_pctchange as line
from dash_multi_page_dashboard.preps_and_tests.unittest_portfolio_balancing_montecarlosim import scatter_plot_ret_vol_sr as scat
from dash_multi_page_dashboard.preps_and_tests.unittest_portfolio_balancing_montecarlosim import bar_plot_sr_all_brackets as brack
from dash_multi_page_dashboard.preps_and_tests.unittest_portfolio_balancing_montecarlosim import barplot_all_sectors_subindustries as bar


import pickle

# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': None,
    'min_rows': None,
    'max_rows': 10,
    'precision': 6,
    'float_format': lambda x: f'{x:.6f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)

currentdatetime = dt.datetime.now()

dash.register_page(__name__, path='/portfolio_optimization', title='Portfolio Allocation and Optimization', name='Portfolio Optimization', order=4)


"""
Colourpicker:
"""

"""
https://encycolorpedia.de/708090
"""

colors = {
    'dark-blue-grey': 'rgb(62, 64, 76)',
    'medium-blue-grey': 'rgb(77, 79, 91)',
    'superdark-green': 'rgb(41, 56, 55)',
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

# Import the file with the tickers and clean it:


def pull_df_from_db_dates_or_not(sql='sp500_adjclose', dates_as_index=False):
    """
    :sql: default is table 'sp500_adjclose' - you can pass any table from the db as argument
    :return:
    """
    connect()
    engine = sqlengine_pull_from_db()

    # an extra integer-index-column is added
    # df = pd.read_sql(sql, con=engine)
    # Column 'Date' is used as index
    if dates_as_index is True:
        df = pd.read_sql(sql, con=engine, index_col='Date', parse_dates=['Date'], )
    else:
        df = pd.read_sql(sql, con=engine)
    return df


def sp500_constituents_allocation_optimization():
    """
    Process:
    1. retrieve in an extra script the current SP500-constituents with beautiful soup (from time to time) (p5_get_sp500_list.py)
     - push table to database; Script: p5_get_sp500list
    2. reset und set new index to sp500-df, slice a current 500-tickers-list from it
    3. we have an old sp500-ratings-list sp500_ratings.xls - we use this list to merge it with the current constituents
    3.a: update the old-xls from time to time if necessary
    4. merge both lists as a main_df - do it in a stand-alone skript: unittest_sp500_...._merged.py
    5. Pull small Df with new ratings from DB (sp500_new_ratings) - this df was constructed and pushed in
    the callback-function add_new_rows here in the dash-main-script - Input via some dbc.buttons or an edit. table
    6. merge the outer_main_df and the new_ratings_df to a main_df_all_ratings and push to DB (you need a Date-col!!!)
    7. Update the empty rating_orig cols with the values of the new ratings directly on DB - copy into a new Table
    with all current ratings - later we can do the update-step via SQLalchemy in the python script
    8. Use this complete table/dataframe (pull from DB) for your optimization-script (so to say, our available stocks)

    :return:
    """

    # df_sp500_verbose = pull_df_from_db(sql='sp500_list_verbose_from_wikipedia')
    """
    Wie genial ist das denn?!? Es läuft reibungslos im Livebetrieb, während Dash app.py läuft,
    konnte ich einfach die Liste durch eine aktuelle ersetzen. :-)
    """
    df_sp500_verbose = pull_df_from_db(sql='sp500_list_verbose_from_wikipedia_2402', dates_as_index=False)

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

    print('Sliced tickers from Dataframe:\n', type(tickers), len(tickers), tickers)

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
    #print(df_ratings)

    # 4. Construct DF from two lists/dataframes:
    # https://realpython.com/pandas-merge-join-and-concat/
    outer_merged_maindf = pd.merge(df_sp500_verbose, df_ratings, on='Symbol', how='outer')
    left_merged_maindf = pd.merge(df_sp500_verbose, df_ratings, on='Symbol', how='left')
    print('##############################')
    print('\n', outer_merged_maindf)
    print('##############################')
    print('\n', left_merged_maindf)
    print('##############################')

    push_df_to_db_replace(left_merged_maindf, 'sp500_leftmerged_old_ratings')

    """
    
    #############################
    
    The merging will happen now server-side in the database....
    
    There we create the table sp500_constituents_updated_ratings
    
    #############################
    
    """

    # Be aware: for merging the DFs into a main_df we need the index, but afterwards we have ro reset_index
    # thus we can use also the Symbol-Column as data-argument in our Dash-Table:
    outer_merged_maindf.reset_index(inplace=True)
    left_merged_maindf.reset_index(inplace=True)

    outer_merged_maindf.to_excel('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500_dfs/main_df_sp500_constituents_ratings.xlsx', engine='openpyxl')
    left_merged_maindf.to_excel('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/investing/sp500_dfs/leftmerged_df_sp500_constituents_ratings.xlsx', engine='openpyxl')

    # retrieve final table and prepare it for final dataframe:
    # df_available_stocks = pull_df_from_db(sql='sp500_constituents_updated_ratings_merged')

    # it is better to retrieve a view...
    df_available_stocks = pull_df_from_db(sql='view_sp500_constituents_updated_ratings')
    #
    #
    #
    #Todo: weiteres von unittest_sp500... übernehmen...erstmal außerhalb lassen...
    #
    #
    #
    return outer_merged_maindf, left_merged_maindf, df_available_stocks


"""    

Data collection    
Cleaning and cleansing

"""
outer_merged, left_merged, available_stocks = sp500_constituents_allocation_optimization()

available_stocks_only_ratings = available_stocks.copy()
available_stocks_only_ratings.dropna(axis=0, subset=['LT-Rating_orig'], inplace=True)
# print('available_stocks_only_ratings:')
# print(available_stocks_only_ratings)

#Todo: for testing purposes we push all necessary df to the DB:
# push_df_to_db_replace(available_stocks, 'sp500_available_stocks')


""" Editing the table - use it as an userinterface for missing ratings"""

constituents_cols = [
    {'id': 'Symbol', 'name': 'Symbol', 'type': 'text', 'format': None},
    {'id': 'security', 'name': 'Security', 'type': 'text'},
    {'id': 'Rating', 'name': 'LT-Rating', 'type': 'text'},
    {'id': 'Name', 'name': 'Name', 'type': 'text'},
    {'id': 'sectors', 'name': 'Sector', 'type': 'text'},
    {'id': 'subindustries', 'name': 'Subindustry', 'type': 'text'},
    {'id': 'headquarters', 'name': 'HQ', 'type': 'text'},
    {'id': 'url', 'name': 'URL', 'type': 'text'},
    {'id': 'added', 'name': 'Added to Index', 'type': 'datetime'},
    {'id': 'founded', 'name': 'Founded', 'type': 'datetime'}
]


available_constituents_cols = [
    {'id': 'Symbol', 'name': 'Symbol', 'type': 'text', 'format': None},
    {'id': 'security', 'name': 'Security', 'type': 'text'},
    {'id': 'LT-Rating_orig', 'name': 'LT-Rating', 'type': 'text'},
    {'id': 'Name', 'name': 'Name', 'type': 'text'},
    {'id': 'sectors', 'name': 'Sector', 'type': 'text'},
    {'id': 'subindustries', 'name': 'Subindustry', 'type': 'text'},
    {'id': 'headquarters', 'name': 'HQ', 'type': 'text'},
    {'id': 'url', 'name': 'URL', 'type': 'text'},
    {'id': 'added', 'name': 'Added to Index', 'type': 'datetime'},
    {'id': 'founded', 'name': 'Founded', 'type': 'datetime'}
]

params2 = ['Symbol', 'LT-Rating', 'Rating-Date', 'Last Review', 'Creditwatch', 'Info']

df_columns = [
    {'id': 'input1', 'name': 'Datetime', 'type': 'datetime'},
    {'id': 'input1', 'name': 'input1', 'type': 'text'},
    {'id': 'input2', 'name': 'input2', 'type': 'text'},
]


columnDefs_sectors = [
    {"field": "Symbol", "sortable": True},
    {"field": "security", "sortable": True},
    {"field": "sectors", "sortable": True},#Sector
    {"field": "subindustries", "sortable": True},
    {"field": "LT-Rating_orig", "sortable": True},
]

####################################################################################################
# 000 - IMPORT DATA

# from the unittest...
####################################################################################################

# replace == in Dash with Dropdown:
df_single_sector_energy = montecarlo.df_all_sectors[montecarlo.df_all_sectors['sectors'] == 'Energy']

df_single_sector_subindustries_grouped = df_single_sector_energy.groupby(['subindustries'], observed=True,
                                                                             as_index=False).count()


sector = 'Energy'
subindustry = 'Integrated Oil & Gas'
subs_stocks_list_energy = []
# df_single_sector_energy = df_all_sectors[df_all_sectors['sectors'] == 'Energy']

df_single_sec_energy = montecarlo.df_all_sectors[(montecarlo.df_all_sectors.sectors == 'Energy')]
# Sort in the right order:
df_single_sec_energy = df_single_sec_energy[['subindustries', 'security']].sort_values(by='subindustries')
aktien_liste_energy = [a for a in df_single_sec_energy['security']]

for j in df_single_sec_energy.subindustries.unique():
    df_single_sub_energy = df_single_sec_energy[df_single_sec_energy.subindustries == j].security.values
    df_single_sub_energy_list = df_single_sub_energy.tolist()
    subs_stocks_list_energy.append(df_single_sub_energy_list)


def main_calculations(rating_bracket=None, n_portfolios: int = 100):
    """
    As data-collection we need two tables:
    1. sp500_constituents_updated_ratings_merged to retrieve only sp-rated stocks (around 430 stocks) and
    2. sp500_adjclose which comprises all pricedata of the sp500 including removed stocks from the index (way more than 500)
    erst einmal alles in eine Funktion schreiben, um nur einmal callen zu müssen
    später dann eine Klasse bilden oder Funktionen auslagern
            ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A']
            ['A-']
            ['BBB+']
            ['BBB']
            ['BBB-']
            ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-']
    :param n_portfolios:
    :param rating_bracket:
    :return:
    """

    if rating_bracket is None:
        rating_bracket = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A']

    # retrieve table and prepare it for final dataframe:
    # pull wo dates:
    df_available_stocks = pull_df_from_db_dates_or_not(sql='sp500_constituents_updated_ratings_merged',
                                                       dates_as_index=False)
    print(f"""
    
        !!!!!!!!!!!!!!!!!!!!!!!
        
        available stocks-DF from main_calulations-function:
        
        {df_available_stocks}
    
        !!!!!!!!!!!!!!!!!!!!!!!
    """)
    ################
    return df_available_stocks










"""
###################
Dash-Layouts:
###################








Ich kann auch erstmal die plotly-Grafiken setzen und
dann entscheiden, ob ich die main_calculations hier neu schreibe 
oder aud der unittest-Datei importiere....



1. erst einmal alle layouts wie im Dashboard-Tutorial bauen


2. vor dem layout schon die Data-Collection, damit die dropdowns mit Options befüllt werden können

3. anstatt die IDs erstmal generische figures, damit ich das Ergebnis sehen kann, danach die figures
aber in den callbacks bauen


"""


def layout():
    """
    Use functions to get reusable code:

    Page layouts must be defined with a variable or function called layout.
    When creating an app with Pages, only use app.layout in your main app.py file.

    Todo: https://stackoverflow.com/questions/73015747/how-do-i-populate-a-dash-datatable-and-append-a-csv-from-user-input
    :return:
    """

    image_path = {'spx_tnx': 'assets/spx_vs_tnx_nov23.PNG',
                  'crown_shyness': 'assets/crown_shyness_rain_forest_canopy_malaysia_deed_3572010709_120c0dda13_o.jpg',
                  'flock_birds': 'assets/Taking_evasive_action_-_geograph.org.uk_-_1069344.jpg',
                  }

    # https://blog.finxter.com/plotly-dash-button-component/

    colors_subs = ['lightsalmon', ] * len(montecarlo.df_sectors_subindustries_grouped.index)

    layout_opti = dbc.Container([
                     dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.H3(children='Portfolio allocation and optimization with S & P 500-constituents',
                                    style={'textAlign': 'center'}),
                            html.Div(children='The idea is, to retrieve always the current S & P 500 constituents. '
                                              'Plot the rating-brackets and returns with plotly. '
                                              'The upper table shows the current SP500-constituents. The lower table '
                                              'shows the for the analysis available stocks, meaning all '
                                              'SP500-constituents with a LT-Rating by Standard and Poors.',
                                     style={'textAlign': 'center'}),
                            html.Br(),
                            html.Hr(),
                        ], width=10),
                        dbc.Col([], width=1),
                        ]),

                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            dash_table.DataTable(
                                data=(left_merged.to_dict(orient='records')),   # chose between left and outer
                                columns=constituents_cols,
                                #columns=(
                                         #[{'id': c, 'name': c} for c in outer_merged.columns]),
                                editable=False,
                                page_size=50,
                                fixed_rows={'headers': True},
                                #page_action='none',
                                style_table={'height': '600px', 'overflowY': 'auto', 'overflowX': 'auto'},
                                filter_action='native',
                                sort_action='native',
                                dropdown={

                                },
                                style_as_list_view=False,
                                style_data_conditional=[],
                                style_data={},
                                style_filter_conditional=[],
                                style_filter={},
                                style_header_conditional=[],
                                style_header={
                                    'backgroundColor': colors['slate-grey-darker'],
                                    'color': 'orange',
                                    'font-size': '1rem',
                                    'fontWeight': 'bold',
                                    'height': 'auto',
                                    #'width': '{}%'.format(len(outer_merged.columns)),
                                    'minWidth': '100px', 'width': '160px', 'maxWidth': '190px',
                                    'whiteSpace': 'normal',
                                    'border': '0px transparent',
                                    'margin': '10px',
                                    'padding': '10px',
                                    'textAlign': 'right'
                                },
                                style_cell_conditional=[],
                                style_cell={
                                    'height': 'auto',
                                    'whiteSpace': 'normal',
                                    #'width': '{}%'.format(len(outer_merged.columns)),
                                    'minWidth': '100px', 'width': '160px', 'maxWidth': '190px',
                                    'margin': '20px',
                                    'padding': '5px',
                                },

                            ),

                        ], width=10),
                        dbc.Col([], width=1),
                    ]),
                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            dash_table.DataTable(
                                data=(available_stocks_only_ratings.to_dict(orient='records')),
                                columns=available_constituents_cols,
                                # columns=(
                                # [{'id': c, 'name': c} for c in outer_merged.columns]),
                                editable=False,
                                page_size=50,
                                fixed_rows={'headers': True},
                                # page_action='none',
                                style_table={'height': '600px', 'overflowY': 'auto', 'overflowX': 'auto'},
                                filter_action='native',
                                sort_action='native',
                                dropdown={

                                },
                                style_as_list_view=False,
                                style_data_conditional=[],
                                style_data={},
                                style_filter_conditional=[],
                                style_filter={},
                                style_header_conditional=[],
                                style_header={
                                    'backgroundColor': colors['slate-grey-darker'],
                                    'color': 'orange',
                                    'font-size': '1rem',
                                    'fontWeight': 'bold',
                                    'height': 'auto',
                                    # 'width': '{}%'.format(len(outer_merged.columns)),
                                    'minWidth': '100px', 'width': '160px', 'maxWidth': '190px',
                                    'whiteSpace': 'normal',
                                    'border': '0px transparent',
                                    'margin': '10px',
                                    'padding': '10px',
                                    'textAlign': 'right'
                                },
                                style_cell_conditional=[],
                                style_cell={
                                    'height': 'auto',
                                    'whiteSpace': 'normal',
                                    # 'width': '{}%'.format(len(outer_merged.columns)),
                                    'minWidth': '100px', 'width': '160px', 'maxWidth': '190px',
                                    'margin': '20px',
                                    'padding': '5px',
                                },

                            ),
                            html.Br(),
                            html.Hr(),
                            html.Plaintext(
                                'Der zweite Table, der die Available Stocks zeigt. \n'
                                'Von hier aus die weitere Analyse starten.\n'
                                'The materialized views are useful in many cases that require fast data access '
                                'therefore they are often used in data warehouses and business intelligence applications.'
                            ),
                        ], width=10),
                        dbc.Col([], width=1),
                    ]),
                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.Br(),
                            html.Hr(),
                            html.Plaintext(
                                'Barchart aller Sektoren zum Auswählen. \n'
                                '\n'
                            ),
                            html.Div('Filters by sector:'),
                            dcc.Dropdown(
                                id='barchart_all_sectors',
                                options=montecarlo.df_sectors_grouped.index,  #['Energy', 'Industrials'], #Todo: replace by list of 11 sectors
                                value=[''],
                                multi=True,
                                placeholder="Select " + 'one Sector' + " (leave blank for all)",
                                style={'font-size': '13px', 'color': colors['slate-grey-darker'], #slate-grey-darker
                                       'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                            ),
                            dcc.Graph(id='', figure=bar()[0]),
                            # Under the first row, the Dash AG grid displays the data from our wine quality dataset. This grid allows
                            # the user to play with the way data is displayed, by moving columns or increasing the numbers of
                            # records displayed at a time.

                            dag.AgGrid(
                                id='grid-sectors',
                                rowData=available_stocks_only_ratings.to_dict("records"),
                                columnDefs=columnDefs_sectors,
                                #columnDefs=[{'field': i} for i in available_stocks_only_ratings.columns],
                                defaultColDef={'resizable': True, 'sortable': True, 'filter': True, 'minWidth': 115},
                                columnSize='sizeToFit',
                                style={"height": "310px"},
                                dashGridOptions={'pagination': True, 'paginationPageSize': 40},
                                className='ag-theme-alpine',
                            ),

                        ], width=10),
                        dbc.Col([], width=1),
                    ]),
                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.Br(),
                            html.Hr(),
                            html.Plaintext(
                                'Barchart aller Subindustries. \n'
                                '\n'
                            ),
                            html.Div('Filters by Subindustry:'),
                            dcc.Dropdown(
                                id='barchart_all_subindustries',
                                options=montecarlo.df_sectors_subindustries_grouped['subindustries'].values   ,#['Transport', 'Apparel'],  # Todo: replace by list of all subindustries
                                value=[''],
                                multi=True,
                                placeholder="Select " + 'one subindustry' + " (leave blank for all)",
                                style={'font-size': '13px', 'color': colors['slate-grey-darker'],  # slate-grey-darker
                                       'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                            ),
                            dcc.Graph(id='', figure=go.Figure(data=[go.Bar(x=montecarlo.df_sectors_subindustries_grouped['subindustries'].values,
                                                                           y=montecarlo.df_sectors_subindustries_grouped.Symbol, marker_color=colors_subs)])),

                        ], width=10),
                        dbc.Col([], width=1),
                    ]),
                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.Br(),
                            html.Hr(),
                            html.Plaintext(
                                'Barchart aller Subs per Sektor. \n'
                                '\n'
                            ),
                            html.Div('Select Sector:'),
                            dcc.Dropdown(
                                id='barchart_subs_per_sector',
                                options=montecarlo.df_sectors_grouped.index,  # Todo: replace by list of subindustries
                                value=[''],
                                multi=True,
                                placeholder="Select " + 'Subindustry' + " (leave blank for all)",
                                style={'font-size': '13px', 'color': colors['slate-grey-darker'],  # slate-grey-darker
                                       'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                            ),
                            dcc.Graph(id='',

                                      figure=go.Figure(data=[
                                          go.Bar(x=df_single_sector_subindustries_grouped.subindustries,
                                                 y=df_single_sector_subindustries_grouped.Symbol,
                                                 hovertext=subs_stocks_list_energy,
                                                 marker_color=colors_subs,
                                                 showlegend=False,
                                                 name='Subs')]),



                                      ),
                        ], width=10),
                        dbc.Col([], width=1),
                    ]),
                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.Br(),
                            html.Hr(),
                            html.Plaintext(
                                'Barchart aller Ratingklammern. \n'
                                '\n'
                            ),
                            html.Div('Select Rating-Bracket or individual selection:'),
                            dcc.Dropdown(
                                id='barchart_rating_brackets',
                                options=['AAA+', 'BBB', 'Individual_1', 'Individual_2'],  # Todo: replace by var rating_bracket and lists of hard coded selections
                                value=[''],
                                multi=True,
                                placeholder="Select " + 'bracket' + " (leave blank for all)",
                                style={'font-size': '13px', 'color': colors['slate-grey-darker'],  # slate-grey-darker
                                       'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                            ),
                            dcc.Graph(id='',
                                      figure=brack()
                                      ),

                        ], width=10),
                        dbc.Col([], width=1),
                    ]),
                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.Br(),
                            html.Hr(),
                            html.Plaintext(
                                'Linechart aller gewählten returns \n'
                                '\n'
                            ),
                            dcc.Graph(id='', figure=line()),

                        ], width=10),
                        dbc.Col([], width=1),
                    ]),
                        dbc.Row([
                            dbc.Col([], width=1),
                            dbc.Col([
                                html.Br(),
                                html.Hr(),
                                html.Plaintext(
                                    'Scatterplot Monte-Carlo-Simulation mit Effizienzlinie. \n'
                                    '\n'
                                ),
                                dcc.Graph(id='', figure=scat()),

                            ], width=10),
                            dbc.Col([], width=1),
                        ]),
                ])

    return layout_opti


"""
Callback: Darstellung der Sektoren - Auswahl je Sektor

und dann alles in die Funktion dieses callbacks einfügen, sodass ich die 
Berechnungen mit intervals und n_clicks über buttons "calculate" steuern kann...
"""


@callback(
    Output(component_id='grid-sectors_graph', component_property='figure'),
    Input(component_id='grid-sectors', component_property='virtualRowData')
)
def grid_sectors(vdata):
    if vdata:
        dff = pd.DataFrame(vdata)
        print(dff)
        figure = go.Figure(data=[go.Bar(x=dff)])
        figure.show()
        return figure
    else:
        figure = go.Figure(data=go.Bar(x=available_stocks_only_ratings.sectors))
        figure.show()
        return figure


#print(montecarlo.df_all_sectors)


if __name__ == '__main__':
    sp500_constituents_allocation_optimization()

