import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output,dash_table, State
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

from scipy.optimize import minimize

from dash.dash_table import FormatTemplate
money = FormatTemplate.money(2)

# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': 30,
    'min_rows': 10,
    'max_rows': 20,
    'precision': 6,
    'float_format': lambda x: f'{x: .6f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)

currentdatetime = dt.datetime.now()

dash.register_page(__name__, path='/montecarlosim', title='8 - Montecarlo-Simulation optimal Portfolio',
                    name='8 - Monte-Carlo-Sim', order=8)

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

####################################################################################################
# 000 - IMPORT DATA - Data collection and doing the main math

# from the unittest...
####################################################################################################


def main_calculations():
    """

    print-Statements are only printed when __name__ == '__main__'

    :return:
    """
    # retrieve table and prepare it for final dataframe:
    # pull wo dates:
    df_available_stocks = pull_df_from_db_dates_or_not(sql='sp500_constituents_updated_ratings_merged',
                                                       dates_as_index=False)

    # print(df_available_stocks)

    avgo = df_available_stocks[df_available_stocks['Symbol'] == 'AVGO']
    print(avgo)

    ind_select = df_available_stocks[df_available_stocks['Symbol'].isin(['NVDA', 'AVGO', 'AAPL', 'MSFT', 'AMAT'])]
    print(ind_select)

    # retrieve the whole df pricedata - dates_as_index=True um einen sortierbaren Index zu haben
    df_pricedata_complete = pull_df_from_db_dates_or_not(sql='sp500_adjclose', dates_as_index=True)

    # Multiindex Date and ID:
    df_pricedata_complete = df_pricedata_complete.reset_index().set_index(['Date', 'ID'])

    # ###### !!!!!!!!!!!! for displaying purposes in AG-Grid we need to see the Date - so just use ID as Index:
    df_pricedata_complete_aggrid = df_pricedata_complete.reset_index().set_index('ID') \
        .sort_values(by='Date', ascending=False)

    return_dictionary = {
        'df_available_stocks': df_available_stocks,
        'df_pricedata_complete': df_pricedata_complete,
        'df_pricedata_complete_aggrid': df_pricedata_complete_aggrid
    }
    return return_dictionary


return_dict = main_calculations()


def layout():
    layout_monte = dbc.Container([
        dbc.Row([
            dbc.Col([], width=1),
            dbc.Col([
                html.H4(children='Montecarlo-Simulation with 10,000 portfolios, efficient frontier and summary table '
                                 'of Sharpe-ratios, weights and optimal portfolio composition',
                        style={'textAlign': 'center'}),
                html.Div(children='The idea is...',
                         style={'textAlign': 'center'}),
                html.Br(),
                html.Hr(),
            ], width=10),
            dbc.Col([], width=1),
        ]),

        dbc.Row([
            dbc.Col([], width=1),
            dbc.Col([
                dcc.Dropdown(id='portfolio-selection',
                             options=[{'label': i, 'value': i} for i in sorted(return_dict['df_available_stocks']['Symbol'].unique())],
                             value=['AVGO', 'AAPL', 'TRGP', 'MKC'],
                             multi=True,
                             placeholder="Select ticker-symbol (leave blank for all)",
                             ),
                html.Br(),
                html.Hr(),

                # Todo: barcharts sectors and Table sectors subindustries

                # Todo: Linechart selection
                dcc.Graph(
                    id='linechart-selection'
                ),
                html.Br(),
                html.Hr(),

                # Todo: Scatterplot selection
                html.Plaintext(
                    'Scatterplot Monte-Carlo-Simulation mit Effizienzlinie. \n'
                    'Choose number of portfolios:'
                    '\n'
                ),
                html.Hr(),
                dcc.RadioItems(id='n-portfolios',
                               options=[
                                   {'label': ' 100', 'value': 100},
                                   {'label': ' 1,000', 'value': 1000},
                                   {'label': ' 10,000', 'value': 10000},
                                   {'label': ' 100,000', 'value': 100000}
                               ],
                               inline=True, value=100,
                               labelStyle={'font-size': 16, 'padding-right': 10, "align-items": "center"},
                               style={'font-size': 16, 'padding-left': 0}), # bezieht sich nur auf die ganze Zeile
                html.Br(),
                html.Hr(),
                dcc.Graph(
                    id='scatterplot-selection',
                    style={'height': '700px'}
                ),

                html.Br(),
                html.Hr(),

                dag.AgGrid(
                    id='summary-table',

                    defaultColDef={'resizable': True, 'sortable': True, 'filter': True, 'minWidth': 115},
                    columnSize='sizeToFit',
                    style={"height": "310px"},
                    dashGridOptions={'pagination': True, 'paginationPageSize': 40},
                    className='ag-theme-balham-dark',
                    # alpine is default - alpine-auto-dark ag-theme-quartz ag-theme-balham-dark
                )


            ], width=10),
            dbc.Col([], width=1),
        ])
    ])  # closes the dbc-Container
    return layout_monte


@callback(
    Output(component_id='linechart-selection', component_property='figure'),
    Input(component_id='portfolio-selection', component_property='value'),
    prevent_initial_call=False
)
def update_input_stocks(values):
    """

    :param values:
    :return:
    """

    #Todo: überlegen hier oder oben adjclose pricedata ziehen??? Was ist less costly???

    df_pricedata_ind_select_ind = return_dict['df_pricedata_complete'][values].sort_index(ascending=True)

    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    df_pricedata_ind_select_ind.query('Date >= @startdate & Date <= @enddate', inplace=True)

    """

    It could be that we won't see a pricedata-graph in case of there are nans within the 200 days - so we fill
    with an "invisible" .001 Dollar Price...

    """

    # Todo: ich muss mit ffill oder bfill arbeiten, um mir mit .001 nicht die return % zu zerschießen :-)
    df_pricedata_ind_select_ind_fillna = df_pricedata_ind_select_ind.bfill()

    # variante filled nans:

    df_pricedata_ind_select_ind_wo_na_no_id = df_pricedata_ind_select_ind_fillna.reset_index().drop(['ID'],
                                                                                                    axis=1).set_index(
        'Date')

    # Todo: Variablen umschreiben - Kraut und Rüben....
    # jeder Tag wird durch den Preis des ersten Tages/erste Zeile geteilt:
    pricedata_normed_ind = df_pricedata_ind_select_ind_wo_na_no_id / df_pricedata_ind_select_ind_wo_na_no_id.iloc[0]

    # -1, damit wir nicht bei 1 starten (also z. B. 180% statt 80% plus)
    df_pricedata_normed_ind = pricedata_normed_ind - 1

    print('pricedata_normed - 1: ', df_pricedata_normed_ind)

    # Ab hier den pricedata-df filtern und mit den nominellen Preisen als options-dict ausgeben:
    dff_pricedata_ind = return_dict['df_pricedata_complete'][values].sort_index(ascending=True)  # ind_select_list

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

    return figure


@callback(
    [Output(component_id='scatterplot-selection', component_property='figure'),
     Output(component_id='summary-table', component_property='rowData'),
     Output(component_id='summary-table', component_property='columnDefs')],
    [Input(component_id='portfolio-selection', component_property='value'),
     Input(component_id='n-portfolios', component_property='value')],
    prevent_initial_call=False
)
def update_scatterplot_montecarlo(values, n_portfolios):
    """

    :return:
    """

    """
        Download daily stock price data per S&P500-group
    """

    # werden etwaige Nans in symbols ein PRoblem? mit bfill füllen? siehe erstes callback oben...
    symbols_list = values

    time_start = dt.datetime.now()

    # slice just the needed columns from pricedata_complete with values/symbols_list:
    df_pricedata_ind_select = return_dict['df_pricedata_complete'][symbols_list].sort_index(ascending=True)

    """
        query eine Datumsklammer
    """
    # startdate = "2023-10-20"
    # enddate = "2024-03-04"
    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    df_pricedata_ind_select.query('Date >= @startdate & Date <= @enddate', inplace=True)

    df_pricedata_ind_select_wo_na = df_pricedata_ind_select.dropna(axis=1)  # row 361

    pricedata_daily_ret_log = np.log(df_pricedata_ind_select_wo_na / df_pricedata_ind_select_wo_na.shift(1)) # row 497
    days = len(pricedata_daily_ret_log)

    print('Symbols list for montecarlo-scat: ', symbols_list)

    print('pricedata_daily_ret_log - w/o Nans: \n', pricedata_daily_ret_log)

    # --------------------------------------
    # Monte-Carlo-Simulation:
    # --------------------------------------
    num_ports = n_portfolios

    all_weights = np.zeros((num_ports, len(symbols_list)))  # Shape is a tuple and yields a matrix num_ports * stocks

    print(f"""
            Number and values of all weights:
            We create a matrix filled with Zeros in {len(all_weights)} rows (n of repetitions) 
            {len(symbols_list)} columns (number of stocks)

            Number: {len(all_weights)}

            Values: {all_weights}
            """)

    # Prepared empty 1D-arrays with Zeros of length of the num-portfolios:
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    # it's more efficient to calculate it once just here instead of in the loop:
    meanLogRet = pricedata_daily_ret_log.mean()
    Sigma = pricedata_daily_ret_log.cov()

    for i in range(num_ports):
        # Create Random Weights between 0 and 1:
        weights_arr = np.array(np.random.rand(len(symbols_list)))

        # Rebalance Weights for that it sums up to 1
        weights_arr = weights_arr / np.sum(weights_arr)
        # print('Weights per portfolio: ', weights_arr)

        # Save weights - geilo, dass ist die übelste Abkürzung für append oder list-comprehension :-)
        # all_weights - right now Zeros - gets populate with the actual weights:
        all_weights[i, :] = weights_arr

        # Expected returns:
        """
        Exkurs: the expected returns (mü - mu) is the secret sauce of Modern Portfolio Theory and CAPM
        It's up to you to calculate it.
        A common way is the CAPM: Riskfree-Rate + Beta * (Rmarket -Rriskfree)
        Beta = cov(asset, benchmark (sp500)) / var(benchmark)
        https://www.youtube.com/watch?v=VsMpw-qnPZY
        
        But in this easy calculation we use as it is common JUST the simple mean of the daily returns:
        """
        ret_arr[i] = np.sum(meanLogRet * weights_arr * days)
        # print('Return of this portfolio: ', ret_arr[i])

        # Expected variance:
        vol_arr[i] = np.sqrt(np.dot(weights_arr.T, np.dot(Sigma * days, weights_arr)))
        # print('Volatility of this portfolio: ', vol_arr[i])

        # Sharpe Ratio
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
        # print('Sharpe ratio of this portfolio: ', sharpe_arr[i])

        pos_weights_maxsr = all_weights[sharpe_arr.argmax(), :]
        all_weights_unstacked = pos_weights_maxsr.tolist()

        #portfolio_best = dict(zip(pos_weights_maxsr, all_weights_unstacked))
        portfolio_best = dict(zip(symbols_list, all_weights_unstacked))

    max_sr = sharpe_arr.max()
    max_ret = ret_arr.max()
    print(f"""
    
         MONTE-CARLO-SIMULATION
        ---------------------------------------------
                All weights from the for-loop for {num_ports} portfolios

                {all_weights}

                all_weights_unstacked:
                {all_weights_unstacked}
        -----------------------------------
        Our max SR after the loop:
        {max_sr}

        The location of the greatest sharpe value in the array of {num_ports} portfolios:
        {sharpe_arr.argmax()}

        Our max return after the loop:
        {max_ret}
        {ret_arr.argmax()}

        These are the desired weights for the portfolio from our Monte Carlo Simulation ONLY for max return:
        {all_weights[ret_arr.argmax(), :]}

        These are the desired weights for the portfolio from our Monte Carlo Simulation (max SR):

        {all_weights[sharpe_arr.argmax(), :]}
        
        That's already a dict with key and values - so we can use it for rowdata :-)
        {portfolio_best}

        -----------------------------------""")

    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
    print('Return at max. Sharpe Ratio: ', max_sr_ret, 'Volatility of max. Sharpe-Ratio: ', max_sr_vol)

    # ------------------------
    # Summary Data-Table - Agg. or Pivottable
    # ------------------------

    df_summary = pd.DataFrame(data=portfolio_best, index=[0])

    # pivot schiebt eine Spalte nach links in den index
    # df_summary = df_summary.pivot(columns=['AAPL'])

    print(df_summary)

    rowData = df_summary.to_dict("records")
    columnDefs = [{'headerName': i, 'field': i} for i in df_summary.columns]

    def get_ret_vol_sr(weights):
        """
        Carrying out a Monte Carlo simulation along with a SciPy minimization function to maximize the overall Sharpe Ratio
        of a certain stock portfolio.
        We can also use SciPy in order to mathematically minimize the negative sharpe ratio,
        giving it's maximum possible SR

        Takes in weights, returns array or return,volatility, sharpe ratio

        https://www.youtube.com/watch?v=f2BCmQBCwDs

        :param weights:
        :return:
        """
        weights = np.array(weights)
        ret = np.sum(pricedata_daily_ret_log.mean() * weights) * days
        vol = np.sqrt(np.dot(weights.T, np.dot(pricedata_daily_ret_log.cov() * days, weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])

    def neg_sharpe(weights):
        """
        SciPy kann nur minimieren, deshalb der Trick mit * -1, um zu maximieren.
        The func get_ret_vol_sr returns an array with ret, vola, sr
        we take index 2 = SR and calculate the neg Sharpe Ratio by multiplying with -1
        :param weights:
        :return:
        """
        return get_ret_vol_sr(weights)[2] * -1

    # Constraints
    def check_sum(weights):
        """
        Returns 0 if sum of weights is 1.0
        :param weights:
        :return:
        """
        return np.sum(weights) - 1

    # By convention of minimize function it should be a function that returns zero for conditions
    cons = ({'type': 'eq', 'fun': check_sum})

    bounds = [(0, 1) for i in range(len(symbols_list))]

    init_guess = [0.0555 for i in range(len(symbols_list) - 1)] + [0.0565]

    # Sequential Least SQuares Programming (SLSQP)
    opt_results = minimize(fun=neg_sharpe, x0=init_guess, method='SLSQP', bounds=bounds, constraints=cons)

    frontier_y = np.linspace(start=0.10, stop=max_ret, num=101)    # Change 101 to a lower number for slower computers!

    def minimize_volatility(weights):
        """

        https://www.youtube.com/watch?v=f2BCmQBCwDs

        :param weights:
        :return:
        """
        return get_ret_vol_sr(weights)[1]

    frontier_volatility = []

    # possible return are all values of an evenly spaced linear-space in its limits like 10 til 30%:
    for possible_return in frontier_y:
        # func for return
        cons = ({'type': 'eq', 'fun': check_sum},
                {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return}
                )
        result = minimize(minimize_volatility, x0=init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        frontier_volatility.append(result['fun'])

    time_stopp = dt.datetime.now()

    elapsed_time = time_stopp - time_start

    print('elapsed_time: ', elapsed_time)

    # Todo: Button n_click für Start Berechnungen und Plotting oder racing bar - progress bar...

    figure_scat = go.Figure(data=go.Scatter(x=vol_arr, y=ret_arr, mode='markers',
                                            marker=dict(
                                                size=12,
                                                color=sharpe_arr,
                                                colorscale='Viridis',
                                                colorbar=dict(title='Sharpe<br>Ratio',
                                                thicknessmode="pixels", thickness=50,
                                                lenmode="pixels", len=400,
                                                yanchor="top", y=.8,
                                                ticks="outside", ticksuffix=" No",
                                                dtick=5
                                                              ),
                                                showscale=True,
                                                line_width=1,
                                                line_color='black'
                                            ),
                                            showlegend=False,
                                            hovertext=[i for i in sharpe_arr],
                                            hoverinfo='text',
                                            ))

    # Add frontier line
    """
    Efficient frontier connects those portfolios which offer the highest expected return for a
    specified level of risk. Portfolios below that line are considered to be sub-optimal.
    """
    figure_scat.add_trace(go.Scatter(x=frontier_volatility, y=frontier_y, mode='lines+markers', name='EF',
                                     marker_color='darkorange'))

    # Add red dot for max Sharpe Ratio
    figure_scat.add_trace(go.Scatter(x=[max_sr_vol], y=[max_sr_ret],
                                     mode='markers',
                                     name='Max SR',
                                     marker=dict(
                                         size=16,
                                         color='darkred',
                                         line_color='black',
                                         line_width=.5
                                     )))

    figure_scat.update_layout(
        font={
            'family': 'Rockwell',
            'size': 16
        },
        # Update title font
        title={
            "text": f"Return and volatility of {len(values)}-stock portfolio for {days} days",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                "family": "Rockwell",
                "size": 20,
                "color": colors['slate-grey-darker']
            }
        }
    )

    figure_scat.update_xaxes(title_text='Volatility of the portfolios')
    figure_scat.update_yaxes(title_text='Return of the portfolios')

    return figure_scat, rowData, columnDefs


if __name__ == '__main__':
    main_calculations()