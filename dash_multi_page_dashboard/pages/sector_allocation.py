import dash

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table, State
import dash_ag_grid as dag

import pandas as pd
import numpy as np

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
    'precision': 6,
    'float_format': lambda x: f'{x:.6f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)

currentdatetime = dt.datetime.now()

dash.register_page(__name__, path='/sector_allocation', title='7 - Portfolio Allocation by Sectors and Subindustries',
                   name='7 - Sector Allocation', order=7)


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
    'light-grey': 'rgb(208, 206, 206)',
    'slate-grey': '#5c6671',
    'slate-grey-invertiert': '#a3998e',
    'slate-grey-darker': '#4b5259',
    'dusk-white': '#CFCFCF',
    'back-white': '#F7F7F7',
    'anthrazitgrau': '#373f43',
    'invertiert': '#c8c0bc',
    'lightsalmon': 'lightsalmon',
    'indianred': 'indianred',
    'darkgoldenrod': 'darkgoldenrod',
    'dodgerblue': 'dodgerblue'

}


# Function for talking with the db:


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


columnDefs_sectors = [
    {"field": "Symbol", "sortable": True},
    {"field": "security", "sortable": True},
    {"field": "sectors", "sortable": True},     # Sector
    {"field": "subindustries", "sortable": True},
    {"field": "LT-Rating_orig", "sortable": True},
]

# allowed dcc.Input types for iterating:
ALLOWED_TYPES = (
    "text",  "search", "range", "hidden",
)

####################################################################################################
# 000 - IMPORT DATA - Data collection and doing the main math

# from the unittest...
####################################################################################################


def main_calculations():
    """
    :return:
    """

    # retrieve table and prepare it for final dataframe:
    # pull wo dates:
    df_available_stocks = pull_df_from_db_dates_or_not(sql='sp500_constituents_updated_ratings_merged',
                                                       dates_as_index=False)

    """
        Variante 1:
        Cleaning and cleansing:
        wir wollen alle, die kein Rating haben, löschen
        it could be that not all sp-constituents have a rating thus we need to construct a dataframe with
        available stocks:
    """

    df = df_available_stocks.copy()
    df.drop(['level_0', 'index'], axis=1, inplace=True)
    df.dropna(axis=0, subset=['LT-Rating_orig'], inplace=True)

    df_complete_all_ratings = df

    # Ticker als DF:
    df_tickers_rating = df_complete_all_ratings[['Symbol', 'LT-Rating_orig', 'security', 'sectors', 'subindustries']]

    """
    #################################################
    Variante 2:
    
    wir beginnen noch einmal mit df_available_stocks und lassen alle ohne Rating drin, weil wir diese später
    im Dash_AG_Grid rausfiltern können.
    
    We wish to aggregate sectors and subindustries and construct filter-lists on the fly:

    #################################################
    """

    df_all_sectors = df_available_stocks.copy()
    df_all_sectors.drop(['level_0', 'index'], axis=1, inplace=True)

    df_all_sectors = df_all_sectors[['Symbol', 'security', 'sectors', 'subindustries', 'LT-Rating_orig']]

    # category-dtype lässt kein groupby zu, wahrscheinlich wegen der Nans; deshalb observed=True:
    # https://medium.com/analytics-vidhya/handling-categories-with-pandas-bfe7d28b2f91
    df_all_sectors = df_all_sectors.astype({'sectors': 'category', 'subindustries': 'category'})

    # we group by the sectors, you need to use a func like count() to show a printable result
    # groupby() is lazy by nature...
    # https://realpython.com/pandas-groupby/#how-pandas-groupby-works
    # use split-apply-combine:
    """
    Handling the categorical attributes in the same way as the textual ones, requires to explicitly set the observed 
    parameter to True in the .groupby method call. Otherwise, pandas will make a cross product of all the available 
    categorical variables used for grouping.
    """

    # use methods like count(), sum(), min(), max(), mean(), median():
    df_sectors_grouped = df_all_sectors.groupby(['sectors'], observed=True)['sectors'].count()

    df_sectors_subindustries_grouped = df_all_sectors.groupby(['sectors', 'subindustries'], observed=True,
                                                              as_index=False).count()
    print('DF grouped sectors and subindustries: \n Attention: the grouped cols are Multi-Index now: \n'
          'You can use as_index=False \n'
          'I think we better should use as_index=False since we later need the cols for Dash... \n',
          df_sectors_subindustries_grouped)

    """
    #####################################################
        Download daily stock price data for S&P500 stocks
        rating_bracket: list with strings:
    #####################################################
    """

    # retrieve the whole df pricedata - dates_as_index=True um einen sortierbaren Index zu haben
    df_pricedata_complete = pull_df_from_db_dates_or_not(sql='sp500_adjclose', dates_as_index=True)

    # Multiindex Date and ID:
    df_pricedata_complete = df_pricedata_complete.reset_index().set_index(['Date', 'ID'])

    # ###### !!!!!!!!!!!! for displaying purposes in AG-Grid we need to see the Date - so just use ID as Index:
    df_pricedata_complete_aggrid = df_pricedata_complete.reset_index().set_index('ID')\
        .sort_values(by='Date', ascending=False)

    """
    
    """
    # Create dropdown options all Symbols
    symbols_unique = df_all_sectors['Symbol'].unique()
    symbol_options_dpdn = [
        {'label': k, 'value': k} for k in sorted(symbols_unique)
    ]

    return_dictionary = {'df_all_sectors': df_all_sectors,
                         'df_sectors_grouped': df_sectors_grouped,
                         'df_sectors_subindustries_grouped': df_sectors_subindustries_grouped,
                         'df_pricedata_complete': df_pricedata_complete,
                         'df_tickers_rating': df_tickers_rating,
                         'df_pricedata_complete_aggrid': df_pricedata_complete_aggrid,
                         'symbol_options_dpdn': symbol_options_dpdn,
                         }

    # return df_all_sectors, df_sectors_grouped, df_sectors_subindustries_grouped, \
        # df_pricedata_complete, df_tickers_rating

    return return_dictionary


#df_all_sectors, df_sectors_grouped, df_sectors_subindustries_grouped, \
    #df_pricedata_complete, df_tickers_rating = main_calculations()

return_dict = main_calculations()


def barplot_all_sectors_subindustries(return_dict):
    """
    df_all_sectors
    :return:
    """

    figure_sec = go.Figure(data=[go.Bar(x=return_dict['df_sectors_grouped'].index,
                                        y=return_dict['df_sectors_grouped'].values,
                                        marker_color=colors['indianred'])])
    # figure.add_trace(go.Bar(x=df_sectors_agg.index, y=df_sectors_agg.values, name='Subindustries'))
    figure_sec.update_layout()
    figure_sec.update_xaxes(title='Fig.4: All sectors')
    figure_sec.update_yaxes()

    df_sectors_subindustries_grouped = return_dict['df_all_sectors'].groupby(['subindustries'], observed=True,
                                                              as_index=False).count()

    figure_all_subs = go.Figure(data=[
        go.Bar(x=df_sectors_subindustries_grouped['subindustries'].values, y=df_sectors_subindustries_grouped.Symbol,
               marker_color=colors['lightsalmon'])])
    figure_all_subs.update_layout()
    figure_all_subs.update_xaxes(title='Fig. 5: All Subindustries')
    figure_all_subs.update_yaxes()


    # https://stackoverflow.com/questions/11869910/pandas-filter-rows-of-dataframe-with-operator-chaining

    return figure_sec, figure_all_subs


figure_sec = barplot_all_sectors_subindustries(return_dict)[0]
figure_all_subs = barplot_all_sectors_subindustries(return_dict)[1]

"""
###################
Dash-Layouts:
###################
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

    layout_opti = dbc.Container([
                     dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.H3(children='Sector allocation with S & P 500-constituents',
                                    style={'textAlign': 'center'}),
                            html.Div(children='The idea is...filtering vrowdata sectors, subs - then'
                                              'extra dcc.Input with collection of max 10 stocks for a portfolio \n '
                                              'zweite Grafik, um die Ind-Select von den Rating- oder Sec.brackets'
                                              'unterscheiden zu können. ',
                                     style={'textAlign': 'center'}),
                            html.Br(),
                            html.Hr(),
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
                            html.Br(),
                            dcc.Graph(id='', figure=figure_sec),    # id is only needed if I populate by callback

                            dag.AgGrid(
                                id='grid-secs-filtered',
                                rowData=return_dict['df_all_sectors'].to_dict("records"),        #available_stocks_only_ratings #df_all_sectors
                                columnDefs=columnDefs_sectors,
                                #columnDefs=[{'field': i} for i in available_stocks_only_ratings.columns],
                                defaultColDef={'resizable': True, 'sortable': True, 'filter': True, 'minWidth': 115},
                                columnSize='sizeToFit',
                                style={"height": "310px"},
                                dashGridOptions={'pagination': True, 'paginationPageSize': 40},
                                className='ag-theme-alpine',
                            ),
                            html.Br(),
                            dcc.Graph(id='barchart-secs'),
                            html.Hr(),
                            html.Br(),





                            dcc.Dropdown(id='symbols-dpdn',
                                         # Create dropdown options all Symbols
                                         options=[{'label': k, 'value': k} for k in sorted(return_dict['df_all_sectors']['Symbol'].unique())],
                                         value=['CAT'],
                                         multi=True,
                                         placeholder="Select ticker-symbol (leave blank for all)",
                                         ),
                            dash_table.DataTable(
                                id='grid-pricedata-filtered',
                                page_size=15,
                                style_table={'width': '100%', 'overflowX': 'auto'},
                                style_header={
                                    'backgroundColor': 'transparent',
                                    'fontFamily': 'Arial',
                                    'font-size': '1rem',
                                    'color': colors['darkgoldenrod'],
                                    'border': '0px transparent',
                                    'textAlign': 'center'},
                                style_cell={
                                    'height': 'auto',
                                    # all three widths are needed
                                    'minWidth': '100px', 'width': '120px', 'maxWidth': '150px',
                                    'whiteSpace': 'normal',
                                    'backgroundColor': 'transparent',
                                    'fontFamily': 'Arial',
                                    'font-size': '0.85rem',
                                    'color': colors['white'],
                                    'border': '0px transparent',
                                    'textAlign': 'center'},
                                style_data={
                                    'color': colors['invertiert'],
                                    'backgroundColor': 'transparent',
                                    'border': '1px solid #a3998e',
                                },
                                cell_selectable=True,
                                column_selectable='multi',
                            ),
                            # dag.AgGrid(
                            #     id='grid-pricedata-filtered',
                            #     # rowData=return_dict['df_pricedata_complete_aggrid'].to_dict("records"),
                            #
                            #     # columnDefs=[{'field': i} for i in return_dict['df_pricedata_complete_aggrid'].columns],
                            #
                            #     defaultColDef={'resizable': True, 'sortable': True, 'filter': True, 'minWidth': 115},
                            #     columnSize='sizeToFit',
                            #     style={"height": "310px"},
                            #     dashGridOptions={'pagination': True, 'paginationPageSize': 40},
                            #     className='ag-theme-alpine-dark',
                            # ),
                            html.Br(),
                            dcc.Graph(id='pricedata-ind'),
                            html.Hr(),
                            html.Br(),

                            # Section: choose your individual stock-selection, out pricedata:
                            # html.Div(
                            #     [dcc.Input(
                            #         id=f'input_{_}',
                            #         type=_,
                            #         placeholder=f"input type {_}"
                            #     ) for _ in ALLOWED_TYPES]
                            #     + [html.Div(id='out-all-types')]
                            # ),
                            html.Br(),
                            # html.Div(
                            #     # debouncing delays the proces of triggering the callback until you hit tab or enter...
                            #     [dcc.Input(id='input-stock1', type='text', value='AAPL', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock2', type='text', value='TDG', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock3', type='text', value='MSFT', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock4', type='search', value='PCAR', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock5', type='search', value='NVDA', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock6', type='search', value='CTRA', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock7', type='search', value='DE', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock8', type='search', value='MRNA', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock9', type='search', value='TRGP', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      dcc.Input(id='input-stock10', type='search', value='COST', placeholder='e.g. TDG',
                            #                debounce=True, autoComplete='on', list='symbols'),
                            #      ]
                            #     + [html.Div(id='out-all-stocks')]
                            # ),
                            html.Br(),
                            html.Datalist(id='symbols', children=[
                                html.Option(value='AWK'),
                                html.Option(value='AMAT'),
                                html.Option(value='ADM'),
                                html.Option(value='CAT'),
                                html.Option(value='EMR'),
                                html.Option(value='HD'),
                                html.Option(value='GILD'),
                                html.Option(value='SWK'),
                                html.Option(value='WMT'),
                                html.Option(value='RL'),
                            ]),
                            dcc.Dropdown(id='symbols-ind-dpdn',
                                         # Create dropdown options individual Symbols
                                         options=[{'label': k, 'value': k} for k in
                                                  sorted(return_dict['df_all_sectors']['Symbol'].unique())],
                                         value=['CAT', 'DE', 'TRGP', 'PCAR', 'MSFT', 'AAPL', 'TDG'],
                                         multi=True,
                                         placeholder="Select ticker-symbol (leave blank for all)",
                                         ),


                            # eventuell den gesamten dag.AgGrid-Block rausnehmen und als var table
                            # unten im Callback bearbeiten, weil von dort der df kommt...
                            # dash_table.DataTable(id='table-ind-pricedata',),
                            dag.AgGrid(
                                #
                                #
                                # wir brauchen nur die Id und das styling - rowData und columnDefs werden aus cb kommen
                                #
                                #
                                #
                                id='table-ind-pricedata',  # pricedata-ind-grid
                                rowData=return_dict['df_pricedata_complete_aggrid'].to_dict("records"),

                                columnDefs=[{'field': i} for i in return_dict['df_pricedata_complete_aggrid'].columns],

                                defaultColDef={'resizable': True, 'sortable': True, 'filter': True, 'minWidth': 115},
                                columnSize='sizeToFit',
                                style={"height": "310px"},
                                dashGridOptions={'pagination': True, 'paginationPageSize': 40},
                                className='ag-theme-balham',
                                # alpine is default - alpine-auto-dark ag-theme-quartz ag-theme-balham-dark
                            ),
                            html.Br(),
                            dcc.Graph(id='pricedata-ind-ind'),
                            html.Hr(),
                            html.Br(),

                            dag.AgGrid(
                                #
                                #
                                # wir brauchen nur die Id und das styling - rowData und columnDefs werden aus cb kommen
                                #
                                #
                                #
                                id='table-prices-aggrid',


                                defaultColDef={'resizable': True, 'sortable': True, 'filter': True, 'minWidth': 115},
                                columnSize='sizeToFit',
                                style={"height": "310px"},
                                dashGridOptions={'pagination': True, 'paginationPageSize': 40},
                                className='ag-theme-balham',
                                # alpine is default - alpine-auto-dark ag-theme-quartz ag-theme-balham-dark
                            ),
                            html.Br(),

                        ], width=10),
                        dbc.Col([], width=1),
                    ])  # closes the dbc.Row
        ])      # closes the dbc.Container

    return layout_opti


"""
Callback: Darstellung der Sektoren - Auswahl je Sektor

und dann alles in die Funktion dieses callbacks einfügen, sodass ich die 
Berechnungen mit intervals und n_clicks über buttons "calculate" steuern kann...

"""


@callback(
    Output(component_id='barchart-secs', component_property='figure'),
    Input(component_id='grid-secs-filtered', component_property='virtualRowData'),
    prevent_initial_call=False,
)
def grid_sectors(vdata):
    if vdata:
        print('vdata: ', vdata)
        dff_single_sec = pd.DataFrame(vdata)
        print(dff_single_sec)
        print(type(dff_single_sec))

        subs_stocks_list_sector = []

        # in Callbacks we use the input: vdata is the already filtered df_single_sec:
        # df_single_sec_energy = df_all_sectors[(df_all_sectors.sectors == 'Energy')]
        # """
        #     df_single_sector_subindustries_grouped goes with Figure 3:
        #     """
        #
        dff_single_sec_subs_grouped = dff_single_sec.groupby(['subindustries'], observed=True,
                                                                                      as_index=False).count()

        # Sort in the right order:
        dff_single_sec = dff_single_sec[['subindustries', 'security', 'Symbol']].sort_values(by='subindustries')
        aktien_liste_energy = [a for a in dff_single_sec['security']]
        print('Aktien_liste aus Listcomprehension: aktien_liste_energy enthält', len(aktien_liste_energy), 'Aktien. ',
              aktien_liste_energy)

        # Erstellen der hoverdata:
        for j in dff_single_sec.subindustries.unique():
            dff_subs = dff_single_sec[dff_single_sec.subindustries == j].security.values
            #print(dff_subs, type(dff_subs))
            df_single_sub_list = dff_subs.tolist()
            #print(df_single_sub_list, type(df_single_sub_list))
            subs_stocks_list_sector.append(df_single_sub_list)

        print('Fertige customdata-Liste nur Energy - subs_stocks_list_energy: ', subs_stocks_list_sector)


        # Sort in the right order:
        dff_symbols = dff_single_sec[['security', 'Symbol']].sort_values(by='Symbol')
        print('DFF_Symbols: ', dff_symbols)

        figure = go.Figure(data=[
            # aus irgendeinem Grund nehme ich hier dff_single_sec_subs_grouped anstatt dff_single_sec ...WARUM???
            go.Bar(x=dff_single_sec_subs_grouped.subindustries, y=dff_single_sec_subs_grouped.Symbol, #warum securities? warum geht Symbol nicht Key-Error Symbol, weil wir oben bei dem sort_values rausslicen OH NEIN
                   hovertext=subs_stocks_list_sector,
                   marker_color=colors['darkgoldenrod'],
                   showlegend=False,
                   )
            ])
        figure.update_layout(barmode='group',
                             hoverlabel=dict(
                                  bgcolor="white",
                                  font_size=16,
                                  font_family="Rockwell"
                                              ),
                             hoverlabel_align='auto',
                             autosize=True,
                             #width=500,
                             height=500,
                             margin=dict(
                                 l=50,
                                 r=50,
                                 b=100,
                                 t=100,
                                 pad=4
                             ),
                             paper_bgcolor="LightSteelBlue",
                            )
        figure.update_xaxes(title='Fig.: Single sector with related Subindustries')
        figure.update_yaxes(title='Count')

        return figure
    else:
        figure = figure_all_subs
        figure.update_layout(barmode='group',
                             hoverlabel=dict(
                                 bgcolor="white",
                                 font_size=16,
                                 font_family="Rockwell"
                             ),
                             hoverlabel_align='auto',
                             autosize=False,
                             # width=500,
                             height=500,
                             margin=dict(
                                 l=50,
                                 r=50,
                                 b=100,
                                 t=100,
                                 pad=4
                             ),
                             paper_bgcolor="LightSteelBlue",
                             )
        figure.update_xaxes(title='Fig.: All Subindustries')
        figure.update_yaxes(title='Count')

        return figure


# Populate the dark pricedata-table by selected symbols from dropdown:
@callback(
    [Output(component_id='grid-pricedata-filtered', component_property='data'),
     Output(component_id='grid-pricedata-filtered', component_property='columns')],
    Input(component_id='symbols-dpdn', component_property='value'),
    prevent_initial_call=False
)
def update_table_dpdn_pricedata(values):
    print('Values es ist eine Liste mit Symbol-strings:', values)
    dff = return_dict['df_pricedata_complete_aggrid'].copy()

    # startdate = "2023-10-20"
    # enddate = "2024-03-04"
    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    dff.query('Date >= @startdate & Date <= @enddate', inplace=True)

    # assign new columns to a DataFrame
    dff = dff.assign(Date=lambda dff: pd.to_datetime(dff['Date'], format='%Y-%m-%d'))

    # This is a simple way to extract the date - NICHT date() verwenden! TypeError: Series object is not callable:
    dff['Date'] = dff['Date'].dt.date

    dff_date_only = dff['Date']

    dff = dff[values]

    # join the date-series and the values-df:
    dff_date_plus_values = dff.join(dff_date_only)

    # print('dff--aggrid-values', dff.head())
    # print('dff--only date', dff_date_only.head())
    print('dff-joined', dff_date_plus_values.head())

    if values:
        data = dff_date_plus_values.to_dict("records")
        columns = [{'id': 'Date', 'name': 'Date'}] + [{'id': i, 'name': i, 'type': 'numeric', 'format': money} for i in values]
# Expected `object`.
    else:
        dfuf= return_dict['df_pricedata_complete_aggrid'].assign(Date=lambda dfuf: pd.to_datetime(dfuf['Date'], format='%Y-%m-%d'))
        dfuf['Date'] = dfuf['Date'].dt.date
        print('Pricedata if "else"')
        print(dfuf.head())
        data = dfuf.to_dict("records")
        columns = [{'id': i, 'name': i, 'type': 'numeric', 'format': money} for i in dfuf.columns]

    return data, columns


@callback(
    Output(component_id='pricedata-ind', component_property='figure'),
    Input(component_id='grid-secs-filtered', component_property='virtualRowData'),
    prevent_initial_call=True
)
def update_pricedata(vdataPrice):
    """
    :param vdataPrice:
    :return:
    """
    #if vdataPrice:
    print('vdata_price: ', vdataPrice)
    dff_select = pd.DataFrame(vdataPrice)
    #print(dff_select)
    #print(type(dff_select))

    # Sort in the right order:
    #dff_symbols = dff_select[['security', 'Symbol']].sort_values(by='Symbol')
    #print('DFF_Symbols: ', dff_symbols)
    #
    ser_ind_select = dff_select['Symbol']
    symbols_list_ind = ser_ind_select.to_list()
    print(f"""

        ser_ind_select:
        {ser_ind_select}

        symbols_list_ind:
        {symbols_list_ind}

    """)

    # slice just the needed columns per rating_bracket with the var symbols_list:
    df_pricedata_ind_select = return_dict['df_pricedata_complete'][symbols_list_ind].sort_index(ascending=True)

    # startdate = "2023-10-20"
    # enddate = "2024-03-04"
    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    df_pricedata_ind_select.query('Date >= @startdate & Date <= @enddate', inplace=True)

    # df_pricedata_ind_select_wo_na = df_pricedata_ind_select.dropna(axis=1)

    # OR without dropping Nans:

    df_pricedata_ind_select_wo_na = df_pricedata_ind_select.copy()

    # variante nans:
    df_pricedata_ind_select_wo_na_no_id = df_pricedata_ind_select_wo_na.reset_index().drop(['ID'], axis=1).set_index('Date')

    pricedata_normed = df_pricedata_ind_select_wo_na_no_id / df_pricedata_ind_select_wo_na_no_id.iloc[0]

    df_pricedata_normed = pricedata_normed - 1

    #print('pricedata_normed - 1: ', df_pricedata_normed)

    figure = go.Figure()
    for col in df_pricedata_normed.columns:
        figure.add_trace(go.Scatter(x=df_pricedata_normed.index,
                                    y=df_pricedata_normed[col], name=col))

    figure.update_layout()

    figure.update_xaxes(title='200-day-bracket',
                                ticks="outside",
                                ticklabelmode="period",
                                tickcolor="black",
                                ticklen=10,
                                minor=dict(
                                    ticklen=4,
                                    dtick=7 * 24 * 60 * 60 * 1000,
                                    tick0="2023-10-01",
                                    griddash='dot',
                                    gridcolor='white')
                                )
    figure.update_yaxes(title='Daily normalized returns')

    return figure
    # else:
    #
    #     # Callback-function for returns-linecharts:
    #
    #     ind_select = ['AAPL', 'MSFT', 'PCAR', 'TRGP', 'TDG', 'TSLA', 'AMAT', 'DE']
    #     df_rating_bracket = df_tickers_rating[df_tickers_rating['Symbol'].isin(ind_select)] \
    #         .sort_values(by='Symbol', ascending=True)
    #     ser_rating_bracket = df_rating_bracket['Symbol']
    #
    #     symbols_list = ser_rating_bracket.to_list()
    #
    #     # slice just the needed columns per rating_bracket with the var symbols_list:
    #     df_pricedata_ind_select = df_pricedata_complete[symbols_list].sort_index(ascending=True)
    #
    #     """
    #         query eine Datumsklammer
    #     """
    #     # startdate = "2023-10-20"
    #     # enddate = "2024-03-04"
    #     timedelta = dt.datetime.now() - dt.timedelta(200)
    #     startdate = timedelta.strftime('%Y-%m-%d')
    #     enddate = dt.datetime.now().strftime('%Y-%m-%d')
    #     df_pricedata_ind_select.query('Date >= @startdate & Date <= @enddate', inplace=True)
    #
    #     """
    #     Drop nans:
    #     """
    #
    #     df_pricedata_ind_select_wo_na = df_pricedata_ind_select.dropna(axis=1)
    #
    #     df_pricedata_ind_select_wo_na_no_id = df_pricedata_ind_select_wo_na.reset_index().drop(['ID'],
    #                                                                                            axis=1).set_index('Date')
    #     pricedata_normed = df_pricedata_ind_select_wo_na_no_id / df_pricedata_ind_select_wo_na_no_id.iloc[0]
    #
    #     # An dieser Stelle müssen wir eine neue symbols_list ohne nans schaffen:
    #     symbols_list_no_nans = df_pricedata_ind_select_wo_na.columns.values.tolist()
    #     print(len(symbols_list_no_nans), symbols_list_no_nans)
    #
    #     df = pricedata_normed - 1
    #
    #     figure = go.Figure()
    #     for col in df.columns:
    #         figure.add_trace(go.Scatter(x=df.index,
    #                                             y=df[col], name=col))
    #     figure.update_layout()
    #
    #     figure.update_xaxes(title='200-day-bracket',
    #                                 ticks="outside",
    #                                 ticklabelmode="period",
    #                                 tickcolor="black",
    #                                 ticklen=10,
    #                                 minor=dict(
    #                                     ticklen=4,
    #                                     dtick=7 * 24 * 60 * 60 * 1000,
    #                                     tick0="2023-10-01",
    #                                     griddash='dot',
    #                                     gridcolor='white')
    #                                 )
    #     figure.update_yaxes(title='Daily normalized returns')
    #
    #     return figure


@callback(
    Output(component_id='out-all-types', component_property='children'),
    [Input(f"input_{_}", "value") for _ in ALLOWED_TYPES]
)
def cb_render(*vals):
    return ' | '.join((str(val) for val in vals if val))


@callback(
    Output(component_id='pricedata-ind-ind', component_property='figure'),
    Input(component_id='symbols-ind-dpdn', component_property='value'),
    prevent_initial_call=False
)
def update_input_stocks(values): # *vals *args, um Schreibarbeit zu sparen; Alternative: input1, input2, input3, input4, input5, input6, input7, input8, input9, input10
    # ind_select_list = [str(val) for val in vals if val]
    # ind_select_list.append([str(val) for val in vals if val])
    # print("""
    # In cb1 sammle ich als Input die 10 Aktien, ich filtere damit den df pricedata
    #ich return den df als list comprehensive dict als output property 'options'
    #""", ind_select_list)

    #output_string = ' | '.join([str(val) for val in vals if val])  # Alternative: {input1} | {input2} | {input3} | {input4} | {input5} | {input6} | {input7} | {input8} | {input9} | {input10}

    df_pricedata_ind_select_ind = return_dict['df_pricedata_complete'][values].sort_index(ascending=True)

    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    df_pricedata_ind_select_ind.query('Date >= @startdate & Date <= @enddate', inplace=True)

    # df_pricedata_ind_select_ind_wo_na = df_pricedata_ind_select_ind.dropna(axis=1)

    # OR without dropping Nans:

    df_pricedata_ind_select_ind_wo_na = df_pricedata_ind_select_ind.copy()

    """
    
    It could be that we won't see a pricedata-graph in case of there are nans within the 200 days - so we fill
    with an "invisible" .001 Dollar Price...
    
    """

    # Todo: ich muss mit ffill oder bfill arbeiten, um mir mit .001 nicht die return % zu zerschießen :-)
    df_pricedata_ind_select_ind_fillna = df_pricedata_ind_select_ind.bfill()

    # variante nans:
    # df_pricedata_ind_select_wo_na_no_id = df_pricedata_ind_select_wo_na.reset_index().drop(['ID'], axis=1).set_index('Date')

    # variante filled nans:

    df_pricedata_ind_select_ind_wo_na_no_id = df_pricedata_ind_select_ind_fillna.reset_index().drop(['ID'], axis=1).set_index(
        'Date')

    # Todo: Variablen umschreiben - Kraut und Rüben....
    pricedata_normed_ind = df_pricedata_ind_select_ind_wo_na_no_id / df_pricedata_ind_select_ind_wo_na_no_id.iloc[0]

    df_pricedata_normed_ind = pricedata_normed_ind - 1

    print('pricedata_normed - 1: ', df_pricedata_normed_ind)

    # Ab hier den pricedata-df filtern und mit den nominellen Preisen als options-dict ausgeben:
    dff_pricedata_ind = return_dict['df_pricedata_complete'][values].sort_index(ascending=True) # ind_select_list

    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    dff_pricedata_ind.query('Date >= @startdate & Date <= @enddate', inplace=True)

    print(dff_pricedata_ind)

    dict_dff_pricedata_ind = [{'label': c, 'value': c} for c in dff_pricedata_ind.columns]
    print(dict_dff_pricedata_ind)

    figure = go.Figure()

    for col in df_pricedata_normed_ind.columns:
        figure.add_trace(go.Scatter(x=df_pricedata_normed_ind.index,
                                    y=df_pricedata_normed_ind[col], name=col))

    return figure #, dict_dff_pricedata_ind output_string


@callback([
     Output(component_id='table-prices-aggrid', component_property='rowData'),
     Output(component_id='table-prices-aggrid', component_property='columnDefs')
    ],
    Input(component_id='symbols-ind-dpdn', component_property='value'),
    prevent_initial_call=False
)
def update_prices_aggrid(values):
    print('Test für letzten Table - Values es ist eine Liste mit Symbol-strings:', values)
    dff = return_dict['df_pricedata_complete_aggrid'].copy()

    df_values = pd.DataFrame(values)
    print('Df_values to get an object: ', df_values)

    # startdate = "2023-10-20"
    # enddate = "2024-03-04"
    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    dff.query('Date >= @startdate & Date <= @enddate', inplace=True)

    # This is a simple way to extract the date - NICHT date() verwenden! TypeError: Series object is not callable:
    dff['Date'] = dff['Date'].dt.date

    dff_date_only = dff['Date']

    dff = dff[values]

    # join the date-series and the values-df:
    # dff_date_plus_values = dff.join(dff_date_only)
    # variante concat
    dff_date_plus_values = pd.concat([dff_date_only, dff], axis=1)

    # print('dff--aggrid-values', dff.head())
    # print('dff--only date', dff_date_only.head())
    print('dff-joined', dff_date_plus_values.head())

    if values:
        # der Fehler array/object lag am Komma hinter der nächsten Zeile - FACEPALM ;-)
        rowData2 = dff_date_plus_values.to_dict("records")
        columnDefs2 = [{'headerName': i, 'field': i} for i in dff_date_plus_values.columns]

    else:
        # bevor wir dt.date zum Abschneiden nutzen können, MÜSSEN wir erst ins Format to_datetime umwandeln!
        dfuf = return_dict['df_pricedata_complete_aggrid'].assign(Date=lambda dfuf: pd.to_datetime(dfuf['Date'], format='%Y-%m-%d'))
        dfuf['Date'] = dfuf['Date'].dt.date
        print('Pricedata if "else"')
        print(dfuf.head())
        rowData2 = dfuf.to_dict("records")
        columnDefs2 = [{'field': i} for i in dfuf.columns]

    return rowData2, columnDefs2


if __name__ == '__main__':
    None