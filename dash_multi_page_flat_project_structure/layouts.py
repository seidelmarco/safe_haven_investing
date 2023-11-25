"""
layouts.py : all pages html layouts will be stored in this file. Given that some components
(like the header or the navbar) have to be replicated on each page, I’ve created some functions to return them,
avoiding many repetitions within the code

All CSS-Bootstrap-grid-system classes can be assigned to the html.Div([]) elements, within their className property.
"""

from dash import Dash, html, dcc, callback, Input, Output, dash_table

import dash_bootstrap_components as dbc

# from dash.dash_table.Format import Format, Group
# import dash_table.FormatTemplate as FormatTemplate

import pandas as pd
import numpy as np

import plotly.graph_objs as go

import plotly.express as px

from datetime import datetime as dt
from dash_multi_page_flat_project_structure.app import app

from investing.p1add_pricedata_to_database import pull_df_from_db

####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### Corporate css formatting

"""
https://encycolorpedia.de/708090
"""

corporate_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',      # header default in tutorial
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(186, 218, 212)',
    'pink-red' : 'rgb(255, 101, 131)',
    'dark-pink-red' : 'rgb(247, 80, 99)',
    'white': 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)',
    'slate-grey': '#5c6671',
    'slate-grey-invertiert': '#a3998e',
    'slate-grey-darker': '#4b5259'

}

externalgraph_rowstyling = {
    #'margin-left': '15px',     # default from tutorial
    #'margin-right': '15px',    # default from tutorial
    'border-color': corporate_colors['slate-grey-darker'],        # default tutorial superdark-green
    'background-color': corporate_colors['slate-grey-darker'],       # default tutorial superdark-green
}

externalgraph_colstyling = {
    'border-radius': '10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': corporate_colors['slate-grey-darker'],        # default tutorial superdark-green
    'background-color': corporate_colors['slate-grey-darker'],       # default tutorial superdark-green
    'box-shadow': '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top': '10px'
}

filterdiv_borderstyling = {
    'border-radius' : '0px 0px 10px 10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['light-green'],
    'background-color' : corporate_colors['light-green'],
    'box-shadow' : '2px 5px 5px 1px rgba(255, 101, 131, .5)'
    }

navbarcurrentpage = {
    'text-decoration': 'underline',
    'text-decoration-color': corporate_colors['slate-grey-invertiert'],     #Tuorial: pink-red
    #'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
    }

recapdiv = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(251, 251, 252, 0.1)',
    'margin-left' : '15px',
    'margin-right' : '15px',
    'margin-top' : '15px',
    'margin-bottom' : '15px',
    'padding-top' : '5px',
    'padding-bottom' : '5px',
    'background-color' : 'rgb(251, 251, 252, 0.1)'
    }

recapdiv_text = {
    'text-align' : 'left',
    'font-weight' : '350',
    'color' : corporate_colors['white'],
    'font-size' : '1.5rem',
    'letter-spacing' : '0.04em'
    }

####################### Corporate chart formatting

corporate_title = {
    'font' : {
        'size' : 16,
        'color' : corporate_colors['white']}
}

corporate_xaxis = {
    'showgrid' : False,
    'linecolor' : corporate_colors['light-grey'],
    'color' : corporate_colors['light-grey'],
    'tickangle' : 315,
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_yaxis = {
    'showgrid' : True,
    'color' : corporate_colors['light-grey'],
    'gridwidth' : 0.5,
    'gridcolor' : corporate_colors['dark-green'],
    'linecolor' : corporate_colors['light-grey'],
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_font_family = 'Dosis'

corporate_legend = {
    'orientation' : 'h',
    'yanchor' : 'bottom',
    'y' : 1.01,
    'xanchor' : 'right',
    'x' : 1.05,
	'font' : {'size' : 9, 'color' : corporate_colors['light-grey']}
} # Legend will be on the top right, above the graph, horizontally

corporate_margins = {'l' : 5, 'r' : 5, 't' : 45, 'b' : 15}  # Set top margin to in case there is a legend

corporate_layout = go.Layout(
    font={'family': corporate_font_family},
    title=corporate_title,
    title_x=0.5,    # Align chart title to center
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=corporate_xaxis,
    yaxis=corporate_yaxis,
    height=270,
    legend=corporate_legend,
    margin=corporate_margins
    )

####################################################################################################
# 000 - DATA MAPPING
####################################################################################################

# Sales mapping
sales_filepath = 'data/datasource.xlsx'

sales_fields = {
    'date': 'Date',
    'reporting_group_l1': 'Country',
    'reporting_group_l2': 'City',
    'sales': 'Sales Units',
    'revenues': 'Revenues',
    'sales_target': 'Sales Targets',
    'rev_target': 'Rev Targets',
    'num_clients': 'nClients'
    }
sales_formats = {
    sales_fields['date']: '%d/%m/%Y'
}


####################################################################################################
# 000 - IMPORT DATA
####################################################################################################

###########################
# Import sales data - mal ein anderer Weg außer read_excel...

file = pd.ExcelFile(sales_filepath)
sales_import = file.parse(sheet_name='Static')
print(type(sales_import))   # result DataFrame
df = sales_import.copy()

# Format date field - dict sales_fields is like our bwIMportAscii for mapping thus we don't need to rename columns:
# .rename() would be another solution
df[sales_fields['date']] = pd.to_datetime(df[sales_fields['date']], format=sales_formats[sales_fields['date']])

# we generate a second date-field with a copy of date - why? what we gonna do with it?
df['date_2'] = df[sales_fields['date']].dt.date
#print(df)
min_dt = df['date_2'].min()
min_dt_str = str(min_dt)
max_dt = df['date_2'].max()
max_dt_str = str(max_dt)
print('Min-Date:', min_dt_str, 'Max-Date:', max_dt_str)

# Create L1 dropdown options
repo_groups_l1 = df[sales_fields['reporting_group_l1']].unique()
repo_groups_l1_all_2 = [
    {'label': k, 'value': k} for k in sorted(repo_groups_l1)
]
print(repo_groups_l1_all_2)
# Result: [{'label': 'Brasil', 'value': 'Brasil'}, {'label': 'Italy', 'value': 'Italy'}, {'label': 'Switzerland', 'value': 'Switzerland'}, {'label': 'USA', 'value': 'USA'}]

repo_groups_l1_all_1 = [{'label': '(Select All)', 'value': 'All'}]
repo_groups_l1_all = repo_groups_l1_all_1 + repo_groups_l1_all_2


# Initialise L2 dropdown options
repo_groups_l2 = df[sales_fields['reporting_group_l2']].unique()
repo_groups_l2_all_2 = [
    {'label': k, 'value': k} for k in sorted(repo_groups_l2)
    ]
repo_groups_l2_all_1 = [{'label': '(Select All)', 'value': 'All'}]
repo_groups_l2_all = repo_groups_l2_all_1 + repo_groups_l2_all_2

"""
Genial: zu jedem Land werden alle passenden Städte als 1:n-Beziehung gemappt. Resultat: ein Dict mit Land +
Array passender Städte.
"""
repo_groups_l1_l2 = {}

# repo_groups_l1 sind alle unique Länder:
for l1 in repo_groups_l1:
    # hier wird jedem Land sein Stadt-Array zugeordnet:
    l2 = df[df[sales_fields['reporting_group_l1']] == l1][sales_fields['reporting_group_l2']].unique()
    print('Array/Liste l2 mit unique Städten:', l2)
    repo_groups_l1_l2[l1] = l2

print(repo_groups_l1_l2)


# import data page 2 (stocks)
def filtered_data_from_df(start_date='2023-10-01', end_date='2023-10-27', ticker: str = 'CTRA'):

    df_stocks = pull_df_from_db(sql='sp500_adjclose')
    df_stocks.reset_index(inplace=True)

    filtered_data = df_stocks.copy()

    # assign new columns to a DataFrame
    filtered_data_newdateformat = filtered_data.assign(Date=lambda filtered_data: pd.to_datetime(filtered_data['Date'], format='%Y-%m-%d'))
    filtered_data_newdateformat.sort_values(by='Date', inplace=True)

    print(filtered_data_newdateformat)
    print(type(filtered_data_newdateformat['Date']))


    filtered_data_loc_one_ticker_all_rows = df_stocks.loc[:, ticker]

    filtered_data_only_ticker_row = df_stocks[ticker]

    return df_stocks, filtered_data, filtered_data_newdateformat, filtered_data_loc_one_ticker_all_rows, filtered_data_only_ticker_row

df_stocks, filtered_data, filtered_data_newdateformat, filtered_data_loc_one_ticker_all_rows, filtered_data_only_ticker_row = filtered_data_from_df()

########################################################################################################## SET UP END

####################################################################################################
# 000 - DEFINE REUSABLE COMPONENTS AS FUNCTIONS
####################################################################################################

#####################
# Header with logo


def get_header():
    """
    We use classNames from the bootstrap.css - am not sure if it wouldn't be sufficient to use dbc instead
    :return:
    """
    header = html.Div([
        html.Div([], className='col-2'),     #Same as img width, allowing to have the title centrally aligned
        html.Div([
            html.H1(children='Performance Dashboard', style={'textAlign': 'center', 'color': corporate_colors['slate-grey-invertiert']}
        )], className='col-8', style={'padding-top': '1%'}
        ),
        html.Div([
            dcc.Link(html.Img(
                src=app.get_asset_url('logo_001c.png'),
                height='43 px',
                width='auto'),
                href='/apps/sales-overview')
        ],
            className='col-2',
            style={
                'align-items': 'center',
                'padding-top': '1%',
                'height': 'auto'}
        )
    ],
            className='row',
            style={'height': '4%',
                   'background-color': corporate_colors['slate-grey-darker']}
    )
    return header


#####################
# Nav bar

def get_navbar(p='sales'):
    """
    comprises all those html-href-hyperlinks:
    :param p:
    :return:
    """
    navbar_sales = html.Div([
        html.Div([], className='col-3'),
        html.Div([
            dcc.Link(
                html.H4(children='Sales Overview',
                        style=navbarcurrentpage),   # Dictionary from the Formatting-section
                href='/apps/sales-overview'
                    )
            ],className='col-2'
            ),

        html.Div([
            dcc.Link(
                html.H4(children='Page 2'),
                href='/apps/page2'
            )
        ],
            className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children='Page 3'),
                href='/apps/page3'
            )
        ],
            className='col-2'),

        html.Div([], className='col-3')

    ],
        className='row',
        style={'background-color': corporate_colors['slate-grey-darker'],  #Tutorial dark-green
               'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    navbar_page2 = html.Div([
        html.Div([], className='col-3'),

        html.Div([
            dcc.Link(
                html.H4(children='Sales Overview'),
                href='/apps/sales-overview'
            )
        ],
            className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children='Page 2',
                        style=navbarcurrentpage),
                href='/apps/page2'
            )
        ],
            className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children='Page 3'),
                href='/apps/page3'
            )
        ],
            className='col-2'),

        html.Div([], className='col-3')

    ],
        className='row',
        style={'background-color': corporate_colors['slate-grey-darker'],
               'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    navbar_page3 = html.Div([
        html.Div([], className='col-3'),

        html.Div([
            dcc.Link(
                html.H4(children='Sales Overview'),
                href='/apps/sales-overview'
            )
        ],
            className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children='Page 2',
                        style=navbarcurrentpage),
                href='/apps/page2'
            )
        ],
            className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children='Page 3',
                        style=navbarcurrentpage),
                href='/apps/page3'
            )
        ],
            className='col-2'),

        html.Div([], className='col-3')

    ],
        className='row',
        style={'background-color': corporate_colors['slate-grey-darker'],
               'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    if p == 'sales':
        return navbar_sales
    elif p == 'page2':
        return navbar_page2
    elif p == 'page3':
        return navbar_page3
    else:
        return navbar_sales


#####################
# Empty row

def get_emptyrow(h='45px'):
    """
    This returns an empty row of a defined height
    :param h:
    :return:
    """

    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className='col-12')
    ],
        className='row',
        style={'height': h,
               'background-color': corporate_colors['slate-grey-darker']
               })

    return emptyrow

####################################################################################################
# Homepage 001 - SALES
####################################################################################################


sales = html.Div([
    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar('sales'),

    #####################
    # Row 3 : Filters
    html.Div([  # External row
        html.Div([   # External 12-column
            html.Div([  # Internal row
                # Internal columns
                html.Div([
                ],
                className='col-2'),     # Blank 2 columns
                # Filter pt.1
                html.Div([
                    html.Div([
                        html.H5(children='Filters by Date:', style={'text-align': 'left', 'color': corporate_colors['medium-blue-grey']}),
                        # Date range picker:
                        html.Div([
                            'Select a date range: ',
                            dcc.DatePickerRange(
                                id='date-picker-sales',
                                start_date=min_dt_str,
                                end_date=max_dt_str,
                                min_date_allowed=min_dt,
                                max_date_allowed=max_dt,
                                start_date_placeholder_text='Start date',
                                display_format='DD-MMM-YYYY',
                                first_day_of_week=1,
                                end_date_placeholder_text='End date',
                                style={'font-size': '12px','display': 'inline-block', 'border-radius': '2px',
                                       'border': '1px solid #ccc', 'color': '#333', 'border-spacing': '0',
                                       'border-collapse':'separate'}
                            )
                        ], style={'margin-top': '5px'})
                    ], style={'margin-top': '10px', 'margin-bottom': '5px', 'text-align': 'left', 'paddingLeft': 5})
                ], className='col-4'),  # End Filter pt 1
                # Filter pt.2
                html.Div([
                    html.Div([
                        html.H5(
                            children='Filters by Reporting Groups:',
                            style={'text-align': 'left', 'color': corporate_colors['medium-blue-grey']}
                        ),
                        # Reporting group selection l1
                        html.Div([
                            dcc.Dropdown(
                                id='reporting-groups-l1dropdown-sales',
                                options=repo_groups_l1_all,
                                value=[''],
                                multi=True,
                                placeholder="Select " + sales_fields['reporting_group_l1'] + " (leave blank for all)",
                                style={'font-size': '13px', 'color': corporate_colors['medium-blue-grey'],
                                       'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                            )
                        ], style={'width': '70%', 'margin-top': '5px'}),
                        # Reporting group selection l2
                        html.Div([
                            dcc.Dropdown(
                                id='reporting-groups-l2dropdown-sales',
                                options=repo_groups_l2_all,
                                value=[''],
                                multi=True,
                                placeholder="Select " + sales_fields['reporting_group_l2'] + " (leave blank for all)",
                                style={'font-size': '13px', 'color': corporate_colors['medium-blue-grey'],
                                       'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                            )
                        ], style={'width': '70%', 'margin-top': '5px'}),
                    ], style={'margin-top': '10px', 'margin-bottom': '5px', 'text-align': 'left', 'paddingLeft': 5})
                ], className='col-4'),  # End Filter pt 2
                html.Div([], className='col-2')     # Blank 2 columns
            ], className='row')     # internal row
        ], className='col-12', style=filterdiv_borderstyling)   # External 12-column
    ], className='row sticky-top'),     # External row

    #####################
    # Row 4 - works as distance-holder between Filters and Graphs:
    get_emptyrow(),

    #####################
    # Row 5 : Charts

    html.Div([      # External row
        html.Div([
        ], className='col-1'),  # Blank 1 column
        html.Div([      # External 10-column
            html.H2(children='Sales Performances', style={'color': corporate_colors['slate-grey-invertiert']}),
            html.Div([      # Internal row - RECAPS
                html.Div([], className='col-4'),    # Empty column
                html.Div([
                    dash_table.DataTable(
                        id='recap-table',
                        style_header={
                            'backgroundColor': 'transparent',
                            'fontFamily': corporate_font_family,
                            'font-size': '1rem',
                            'color': corporate_colors['light-green'],
                            'border': '0px transparent',
                            'textAlign': 'center'},
                        style_cell={
                            'backgroundColor': 'transparent',
                            'fontFamily': corporate_font_family,
                            'font-size': '0.85rem',
                            'color': corporate_colors['white'],
                            'border': '0px transparent',
                            'textAlign': 'center'},
                        cell_selectable=False,
                        column_selectable=False
                    )
                ], className='col-4'),
                html.Div([], className='col-4')     # Empty column
            ], className='row', style=recapdiv),    # Internal row - RECAPS
            html.Div([      # internal row
                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='sales-count-day'
                    )
                ], className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='sales-count-month')
                ], className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='sales-weekly-heatmap')
                ], className='col-4')
            ],
            className='row'),   # Internal row

            html.Div([ # Internal row

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='sales-count-country')
                ], className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='sales-bubble-county')
                ], className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='sales-count-city')
                ], className='col-4')

            ], className='row')   # Internal row


            ], className='col-10', style=externalgraph_colstyling),  # External 10-column
            html.Div([], className='col-1'),    # Blank 1 column
        ], className='row', style=externalgraph_rowstyling),     # External row
    ])


####################################################################################################
# 002 - Page 2 - stock analysis
####################################################################################################

tickers = df_stocks.columns
tickers = tickers[1:]
markets = ['SP500', 'Eurostoxx', 'Top200_Africa']

page2 = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('page2'),


    #####################
    #Row 3 : Filters
    html.Div([ # External row

        html.Br()

    ], className='row sticky-top'),   # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts


    html.Div([  # External row

        html.Br(),
        html.H4(children='This is page 2'),
        html.P(children='Was will ich darstellen: auf Postgres-DB schauen und rausschreiben, was ich sehen will - '
                        'unbedingt ein AGGrid für alle SP500-Zeilen einbauen. Was will ich wie filtern? '
                        'Als Start- und Enddatum nehme ich ersten und letzten DS aus sp500_adjclose.'),
        html.Br(),


        # divisions for dropdowns:


        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Markets", className="menu-title"),
                        dcc.Dropdown(
                            id="market-filter",
                            options=[
                                {"label": market, "value": market}
                                for market in markets
                            ],
                            value="SP500",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
        html.Br(),
        html.Div(
                    children=[
                        html.Div(children="Ticker", className="menu-title"),
                        #dcc.Dropdown(
                            #id="ticker",
                            #options=[
                                #{"label": ticker, "value": ticker}
                                #for ticker in tickers
                            #],
                            #value="MSFT",
                            #clearable=False,

                            #className="dropdown",
                        #),
                        dcc.Dropdown(options=df_stocks.columns, value='MSFT', id='ticker', clearable=False, className='dropdown'),
                    ]
                ),
        html.Br(),
        html.Div([
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df_stocks['Date'].min().date(),
                max_date_allowed=df_stocks['Date'].max().date(),
                start_date=df_stocks['Date'].min().date(),
                end_date=df_stocks['Date'].max().date(),
                start_date_placeholder_text=df_stocks['Date'].min().date(),
                end_date_placeholder_text=df_stocks['Date'].max().date(),
                #start_date='2023-01-01',
                #end_date='2023-12-31',
            )
        ])
            ], #className='menu'
        ),

        html.Br(),
        html.Div([
            dcc.Graph(
                id='adjclose_chart',
            )
        ], className='card'),     # .card, .wrapper comes from 003_style.css - col-8 in case of Bootstrap-usage
        html.Br(),
        html.Div([
            dcc.Graph(
                responsive='auto'
            )
        ], className='card'),
        html.Br(),
        html.Div([
            dcc.Graph(

            )
        ], className='card')
    ], className='wrapper')

])

####################################################################################################
# 003 - Page 3 - empty page
####################################################################################################

page3 = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('page3'),

    #####################
    #Row 3 : Filters
    html.Div([  # External row

        html.Br()

    ], className='row sticky-top'),     # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([  # External row

        html.Br(),
        html.H4(children='This is page 3'),

    ])
])

if __name__ == '__main__':
    stock_data = pull_df_from_db(sql='sp500_adjclose')
    print(stock_data.sort_values(by='Date'))
    print(tickers)
    print(type(tickers))