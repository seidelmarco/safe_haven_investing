import dash

import dash_bootstrap_components as dbc

from dash import Dash, dash_table, dcc, html, Input, Output, callback, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go

# For SQLAlchemy
import sqlalchemy as db
from sqlalchemy import create_engine, ForeignKey, Column, String, Integer, CHAR, CheckConstraint, join, Date, text

# Documentation: earthly.dev/ (sqlalchemy)
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import sessionmaker, relationship


import pandas as pd

import yfinance as yf

import datetime as dt


from investing.p5_get_sp500_list import save_sp500_tickers

from investing.myutils import connect, sqlengine, sqlengine_pull_from_db, talk_to_me

timestamp = dt.datetime.now()


dash.register_page(__name__, path='/editable', title='Editable Table', name='Editable Table', order=5)


params = [
    'Symbol', 'Security', 'LT-Rating', 'Rating-Date', 'Last Review',
    'Creditwatch/Outlook', 'Remarks'
]

"""
#############
Colourpicker:
#############
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
####################
UTILS
####################
"""


def pull_df_from_db_dash(sql='sp500_adjclose'):
    """
    # Todo : Funktion umschreiben, dass ich aus allen Tabellen und ausgewählte Spalten beim callen wählen kann
    :sql: default is table 'sp500_adjclose' - you can pass any table from the db as argument
    :return:
    """
    connect()
    engine = sqlengine_pull_from_db(future=True)

    df = pd.read_sql(sql, con=engine)

    return df


def push_df_to_db_replace_dash(df, tablename: str):
    """
    You can use talk_to_me() or connect()
    :tablename: str
    :return:
    """

    # talk_to_me()

    connect()

    engine = sqlengine(future=True)

    # Todo: how to inherit if_exists to push_df_to_db-function?
    df.to_sql(tablename, con=engine, if_exists='replace', chunksize=100, index=False)


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
    {'id': 'symbol', 'name': 'Symbol', 'type': 'text', 'format': None},
    {'id': 'security', 'name': 'Security', 'type': 'text'},
    {'id': 'rating_orig', 'name': 'LT-Rating', 'type': 'text'},
    {'id': 'Name', 'name': 'Name', 'type': 'text'},
    {'id': 'sectors', 'name': 'Sector', 'type': 'text'},
    {'id': 'subindustries', 'name': 'Subindustry', 'type': 'text'},
    {'id': 'headquarters', 'name': 'HQ', 'type': 'text'},
    {'id': 'url', 'name': 'URL', 'type': 'text'},
    {'id': 'added', 'name': 'Added to Index', 'type': 'datetime'},
    {'id': 'founded', 'name': 'Founded', 'type': 'datetime'}
]

params2 = ['Symbol', 'Rating', 'Rating-Date', 'Last Review', 'Creditwatch', 'Info']

df_columns = [
    {'id': 'input1', 'name': 'Datetime', 'type': 'datetime'},
    {'id': 'input1', 'name': 'input1', 'type': 'text'},
    {'id': 'input2', 'name': 'input2', 'type': 'text'},
]



"""
####################
LAYOUT
####################
"""


def layout():
    """

    :return:
    """

    layout_editable = dbc.Container([
        dbc.Row([
            dbc.Col([], width=1),
            dbc.Col([
                html.H3(children='Testing an editable table',
                        style={'textAlign': 'center'}),
                html.Div(children='The idea is, we construct a table with predefined cols (list of params)\n'
                                  '- sp-ratings erfassen\n'
                                  '- Date im Format 2024-01-01 erfassen - wird auf der DB umgewandelt oder timestamp vorbesetzen\n'
                                  '- upload nur nach n_clicks > 0 ... mit State arbeiten --- die Idee ist, anstatt'
                                  'einzelne Input-Felder EINE Tabellenzeile als Input- und Output-Form zu nutzen',
                         style={'textAlign': 'center'}),
                html.Plaintext('1.  From page portfolio opti. : Table with SP500 constituents; wo trage ich meine Ratings ein?'
                               ' Schon in der fertig gemergten?', style={'color': 'green'}),
                html.Plaintext('The 1. idea: look at the form of the Input-table...SELECT * FROM public.sp500_new_ratings; ',
                               style={'color': 'green'}),
                html.Plaintext('SELECT * FROM public.sp500_new_ratings: index, Date, symbol, rating, rating_date, last_review, creditwatch, info',
                               style={'color': 'green'}),
                html.Plaintext('2. Firstly, I populate by hand the sp500_new_ratings. It seems, that this table isnt gonna used yet',
                               style={'color': 'green'}),
                html.Plaintext('3. Since I am able to delete rows and columns in sp500_new_ratings I HAVE TO merge it'
                               'server-side with the main-table ... so that new data is stored persistently! ',
                               style={'color': 'green'}),
                html.Br(),
                html.Hr(),
                #Todo: INSERT
                html.Div([
                    html.Br(),
                    dcc.Input(id='adding-rows-name', placeholder='Enter a column name...', value='', type='text',
                              style={'padding': 10}),
                    html.Button(children='Add Column', id='adding-columns-button', n_clicks=0, n_clicks_timestamp=-1,
                                style={'padding': 10})
                ], style={'height': 'auto'}),
                html.Br(),
                html.Br(),

                # activated once/week or when page refreshed - 1 day in ms * 7
                dcc.Interval(id='interval_pg', interval=86400000 * 7, n_intervals=0),

                # the whole table gonna be just an empty list - later populated by the callback...
                html.Div(id='postgres_datatable', children=[]),
                html.Br(),
                html.Button('Add Row', id='editing-rows-button', n_clicks=0),
                html.Button('Export xls and Upload DB', id='save_to_postgres', n_clicks=0),
                html.Br(),
                html.Br(),
                html.Hr(),

                # Create notification when saving to excel
                # next 3 lines are just generic syntax for a default notification
                html.Div(id='placeholder_edi_table', children=[]),
                dcc.Store(id='store_edi_table', data=0),
                dcc.Interval(id='interval', interval=2000),

                # you see an empty generic figure-dict since we populate via the callback:
                dcc.Graph(id='my_graph', figure={'data': [], 'layout': {}, 'frames': []})
                # ----------------------------------------------------------------------------------
            ], width=10),
            dbc.Col([], width=1),
        ]),

        dbc.Row([
            dbc.Col([], width=1),
            dbc.Col([
                #html.Div(children="That's the output table - populate with the data from above table and use States for updating...",
                 #        style={'textAlign': 'center'}),
                #html.Br(),
                #html.Hr(),
                #html.Br(),
                #dash_table.DataTable(
                        #id='update-output-table',
                    #data=df_output.to_dict(orient='records'), #[]
                    #columns=([{'id': 'Entry', 'name': 'Entry'}] +
                    #         [{'id': 'Date', 'name': 'Date'}] +
                    #         [{'id': p, 'name': p} for p in params]),
                    # HOW TO USE THE INPUT AS OUTPUT - STORE DURING SESSION?
                    #editable=True,
                    #)
            ], width=10),
            dbc.Col([], width=1),
            ]),         # Ende Reihe mit Output Table
    ])  # Ende Layout Container

    return layout_editable



"""
####################
CALLBACKS/FUNCTIONS
####################
"""


"""
What is the difference between state and input in dash?
Inputs will trigger your callback; State do not. If you need the the current “value” - aka State - 
of other dash components within your callback, you pass them along via State .

See https://dash.plot.ly/state 4.1k for some examples.
"""


@callback(
    Output('postgres_datatable', 'children'),
    [Input('interval_pg', 'n_intervals')])
    # interval_pg ist das mit einer Woche - ohne zu klicken brauchen wir das Laden des Tables nur einmal wöchentlich...
def populate_datatable(n_intervals):
    df = pull_df_from_db_dash(sql='sp500_new_ratings')
    return [
        dash_table.DataTable(
            id='new_ratings_table',
            columns=[{
                'name': str(x),
                'id': str(x),
                'deletable': False,
            } if x == 'Symbol' or x == 'Security' or x == 'LT-Rating'
              else {
                'name': str(x),
                'id': str(x),
                'deletable': True,
            } for x in df.columns],
            data=df.to_dict('records'),
            editable=True,
            page_size=50,
            fixed_rows={'headers': True},
            row_deletable=True,
            filter_action="native",
            sort_action="native",  # give user capability to sort columns
            sort_mode="single",  # sort across 'multi' or 'single' columns
            page_action='none',  # render all the data at once. No paging.
            style_table={'height': '600px', 'overflowY': 'auto', 'overflowX': 'auto'},
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
            style_cell={
                'textAlign': 'left',
                'height': 'auto',
                'whiteSpace': 'normal',
                # 'width': '{}%'.format(len(outer_merged.columns)),
                'minWidth': '100px', 'width': '160px', 'maxWidth': '190px',
                'margin': '20px',
                'padding': '5px',
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'right'
                } for c in ['Rating-Date', 'Last Review']
            ]
        ),
    ]


@callback(
    Output(component_id='new_ratings_table', component_property='columns'),
    [Input(component_id='adding-columns-button', component_property='n_clicks')],
    [State(component_id='adding-rows-name', component_property='value'),
     State(component_id='new_ratings_table', component_property='columns')],
    prevent_initial_call=True)
def add_columns(n_klicks, value, existing_columns):
    print(existing_columns)
    if n_klicks > 0:
        existing_columns.append({
            'name': value, 'id': value,
            'renamable': True, 'deletable': True
        })
    print(existing_columns)
    return existing_columns


@callback(
    Output(component_id='new_ratings_table', component_property='data'),
    [Input(component_id='editing-rows-button', component_property='n_clicks')],
    [State(component_id='new_ratings_table', component_property='data'),
     State(component_id='new_ratings_table', component_property='columns')],
    prevent_initial_call=True)
def add_row(n_klicks, rows, columns):
    print(rows)
    if n_klicks > 0:
        rows.append({
            c['id']: '' for c in columns
        })
    print(rows)
    return rows


@callback(
    Output('my_graph', 'figure'),
    [Input('new_ratings_table', 'data')],
    prevent_initial_call=True,
)
def display_graph(data):
    df_fig = pd.DataFrame(data)
    print(f"""
            Dataframe df_fig...
         {df_fig}df_fig""")

    df_groups = df_fig.groupby(df_fig['LT-Rating']).agg('sum')
    print(df_groups)
    # fig = px.bar(df_fig, x='LT-Rating', y='Security')
    fig = px.histogram(df_fig, x='LT-Rating', barmode='group',
                       category_orders=dict(df_fig=['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-']),
                       color='Symbol', color_discrete_sequence=['darkgoldenrod'], color_discrete_map={}
                       )
    fig.update_layout()

    # Todo: Alternative....
    fig_alt = px.bar(
            df_fig.groupby("LT-Rating", as_index=False)
            .agg(func="count")
            .sort_values(["LT-Rating"], ascending=[0]),
            x="LT-Rating",
            y="Symbol",
            hover_data=["Symbol"],
            title="Most Common Words that Speakers Use",
            ),

    # figure_go = go.Figure() - minimal Statement um eine leere Grafik ohne Crash zu sehen...
    figure_go = go.Figure(
        data=[go.Bar(
            x=['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-'],
            y=df_fig['LT-Rating'].value_counts(),
            marker=go.bar.Marker(autocolorscale=True, color='orange'),
            hovertemplate='<br>'.join(
                ['Rating class: %{x}', 'No. of constituents: %{y}'])
            )])

    # the fig goes into the 'my_graph' 'figure' - it's an empty parameter above
    # you can leave blank nearly every section in the layout - you can design everything here in the callbacks
    # and via the id you write back into the layout...

    return fig #figure_go


@callback(
    [Output('placeholder_edi_table', 'children'),
     Output('store_edi_table', 'data')],
    [Input('save_to_postgres', 'n_clicks'),
     # der Outputtable soll alle zwei Sekunden neu geladen werden - deshalb feuern wir den callback mit interval...
     Input('interval', 'n_intervals')],
    [State('new_ratings_table', 'data'),
     State('store_edi_table', 'data')],
    prevent_initial_call=True)
def df_to_db(n_clicks, n_intervals, dataset, s):
    output = html.Plaintext('The data has been saved to your folder and DB.',
                            style={'color': 'green', 'font-weight': 'bold', 'font-size': 'large'})
    no_output = html.Plaintext('', style={'margin': '0px'})

    input_triggered = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if input_triggered == 'save_to_postgres':
        s = 6
        df = pd.DataFrame(dataset)
        df.to_csv('sp500_new_ratings.csv')
        push_df_to_db_replace_dash(df=df, tablename='sp500_new_ratings')
        return output, s
    elif input_triggered == 'interval' and s > 0:
        s = s-1
        if s > 0:
            return output, s
        else:
            return no_output, s
    elif s == 0:
        return no_output, s








