from urllib.request import Request, urlopen
import json
import ssl
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from dash_multi_page_dashboard.preps_and_tests.unittest_returns_table import normalized_returns_from_database

import pandas as pd
import pandas_datareader as web
from pandas.api.types import CategoricalDtype
from tqdm import tqdm

# Bsp. für alle Schleifen/Iterables - List comprehensions:
# sharpe_ratios, wghts = np.column_stack([sharpe_ratios_and_weights(returns) for x in tqdm(range(n))])

import pprint
import datetime as dt
from investing.myutils import timestamp, sqlengine_pull_from_db
import investing.hidden
from investing.connect import connect

import pickle

from investing.p1add_pricedata_to_database import pull_df_from_db

from investing.p5_get_sp500_list import save_sp500_tickers

from openpyxl import writer
import openpyxl

settings = {
    'max_columns': 10,
    'min_rows': None,
    'max_rows': 10,
    'precision': 6,
    'float_format': lambda x: f'{x:.6f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)



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


# Import the file with the tickers and clean it:


def pull_df_from_db_dates_or_not(sql='sp500_adjclose', dates_as_index=False):
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
    if dates_as_index is True:
        df = pd.read_sql(sql, con=engine, index_col='Date', parse_dates=['Date'], )
    else:
        df = pd.read_sql(sql, con=engine)
    return df


def get_current_tickerlist_sp500(*args, **kwargs):
    """
    Fallback-Funktionen, wenn wir nicht die
    'sp500_list_verbose_from_wikipedia_2402' von der DB haben:

    df_sp500_verbose = save_sp500_tickers(edit=False)['df']
    df_sp500_verbose.index.name = 'Symbol'
    :return:
    """

    # retrieve table and prepare it for final dataframe:
    # pull wo dates:
    df_available_stocks = pull_df_from_db_dates_or_not(sql='sp500_constituents_updated_ratings_merged',
                                                       dates_as_index=False)

    print(f"""
        ------------------------------------------------------------------
        df_available_stocks from the Database - all sp500 even no ratings:
        ------------------------------------------------------------------
        {df_available_stocks[['Symbol', 'security', 'LT-Rating_orig', 'sectors', 'subindustries']]}
    """)

    #deadend00 = input('Deadend00 - Press enter to run next steps...')

    """
    Cleaning and cleansing:
    wir wollen alle, die kein Rating haben, löschen
    it could be that not all sp-constituents have a rating thus we need to construct a dataframe with
    available stocks:
    """
    # print(df_available_stocks.columns)
    # print(df_available_stocks.describe(include='all'))
    df = df_available_stocks.copy()
    df.drop(['level_0', 'index'], axis=1, inplace=True)
    df.dropna(axis=0, subset=['LT-Rating_orig'], inplace=True)
    print(f"""
            ------------------------------------------------------------------
            df cleansed w/o nan-rows  - all sp500 {df.shape} only with ratings:
            ------------------------------------------------------------------
            {df[['Symbol', 'security', 'LT-Rating_orig', 'sectors', 'subindustries']]}
        """)
    # print(df.describe(include='all'))

    #deadend0 = input('Deadend0 - Press enter to run next steps...')

    # securities with rating as tickers-list:
    tickers_rating_as_list = df['Symbol'].tolist()

    print('Sliced tickers from Dataframe - only tickers with rating; '
          'you find them in the datacollector as tickers_rating_as_list:\n',
          type(tickers_rating_as_list), len(tickers_rating_as_list),
          tickers_rating_as_list)

    # For checking:
    # print(df_available_stocks.loc[df_available_stocks['Symbol'] == 'AOS'])

    df_complete_all_ratings = df

    # Ticker als DF:
    df_tickers_rating = df[['Symbol', 'LT-Rating_orig', 'security']]
    # print(df_tickers_rating)

    """

    Diese Zeilen verstehe ich nach wie vor nicht:
    stocks_w_rating = list(returns.columns)
    tickers_available = df_available_stocks['Symbol'][df_available_stocks['Symbol'].str.match(('|'.join(stocks_w_rating)))].dropna()
    print('')
    print('Tickers available w/o NA:')
    print(tickers_available)
    """

    return {'df_available_stocks': df_available_stocks,
            'df_complete_all_ratings': df_complete_all_ratings,
            'tickers_rating_as_list': tickers_rating_as_list,
            'df_tickers_rating': df_tickers_rating}


global data_collector

data_collector = get_current_tickerlist_sp500()


def pricedata_per_ratingbracket(rating_bracket=None):
    """
    Download daily stock price data for S&P500 stocks
    rating_bracket: list with strings:
        ['A-']
        ['BBB+']
        ['BBB']
        ['BBB-']
        ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-']
    :return:
    """

    if rating_bracket is None:
        rating_bracket = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A']

    df_complete_all_ratings = data_collector['df_complete_all_ratings']
    df_tickers_rating = data_collector['df_tickers_rating']

    print(f""" 
        --------------------------------------------------------------------------
        Dataframe Tickers/Securities with rating Aus Funktion pricedata_per_ratingbracket:
        you must see all available 445 Symbols, Rating_orig and security
        --------------------------------------------------------------------------
        """)
    print(df_tickers_rating)

    # for testing:
    # df_rating_bracket_scalar = df_tickers_rating[df_tickers_rating['LT-Rating_orig'] == 'AAA']
    # print(df_rating_bracket_scalar)

    # deadend1 = input('Next step...press Enter...')

    """ Either... """
    df_rating_bracket = df_tickers_rating[(df_tickers_rating['LT-Rating_orig'] == 'AAA') |
                                 (df_tickers_rating['LT-Rating_orig'] == 'AA+') |
                                 (df_tickers_rating['LT-Rating_orig'] == 'AA') |
                                 (df_tickers_rating['LT-Rating_orig'] == 'AA-')
                                ]
    # .values returns lists in a list :-)                           ]
    #df_group = df_tickers_rating[
        #(df_tickers_rating['LT-Rating_orig'] == 'AAA') | (df_tickers_rating['LT-Rating_orig'] == 'AA+')].values

    """ 
    
    Or... 
    
    """
    df_rating_bracket = df_tickers_rating[df_tickers_rating['LT-Rating_orig'].isin(rating_bracket)]\
        .sort_values(by='LT-Rating_orig', ascending=True)
    print(f""" 
            --------------------------------------------------------------------------
            Dataframe Tickers/Securities with rating Aus Funktion pricedata_per_ratingbracket:
            you must see 82(hier noch curly var einbauen)-Elements for the bracket {rating_bracket}
            --------------------------------------------------------------------------
            """)
    print(df_rating_bracket)

    # deadend2 = input('Next step...press Enter...')

    """
    Download daily stock price data per S&P500-group
    """
    df_1col = df_rating_bracket
    ser_symbols = df_1col['Symbol']
    del df_1col
    # print(ser_symbols)
    print(f""" 
                --------------------------------------------------------------------------
                You see 82(hier noch curly var einbauen)- Series of Symbols/tickers-shape for the bracket {rating_bracket}
                --------------------------------------------------------------------------
                """)
    print(ser_symbols.shape)
    print(ser_symbols.describe())

    symbols_list = ser_symbols.to_list()
    print(symbols_list)

    # retrieve the whole df pricedata - dates_as_index=True um einen sortierbaren Index zu haben
    df_pricedata = pull_df_from_db_dates_or_not(sql='sp500_adjclose', dates_as_index=True)
    # print(df_pricedata.columns.values)

    # Multiindex Date and ID:

    df_pricedata = df_pricedata.reset_index().set_index(['Date', 'ID'])

    # slice just the needed columns per rating_bracket with the var symbols_list:
    df_pricedata_per_rating_bracket = df_pricedata[symbols_list].sort_index(ascending=True)

    """
        query eine Datumsklammer
    """
    startdate = "2024-01-01"
    #enddate = "2024-03-04"
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    df_pricedata_per_rating_bracket.query('Date >= @startdate & Date <= @enddate', inplace=True)

    print(df_pricedata_per_rating_bracket)

    # deadend3 = input('Next step...press Enter...')

    """
    Drop nans:
    """

    df_pricedata_per_group_wo_na = df_pricedata_per_rating_bracket.dropna(axis=1)

    """
    Transform the price matrix into a return matrix:
    𝑅𝑒𝑡𝑢𝑟𝑛𝑖,𝑇 = 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇 − 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1/𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 = 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇/𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 − 1 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1
    shift(1) geht eine Zeile zurück (schiebt eine Zeile von oben auf den cursor (die aktuelle loop))
    """

    # Returns stimmen - mit XOM geprüft:
    df_returns_per_rating_bracket = (df_pricedata_per_group_wo_na/df_pricedata_per_group_wo_na.shift(1) - 1)[1:]

    print(f""" 
                    --------------------------------------------------------------------------
                    You see 82(hier noch curly var einbauen)- daily returns for the bracket {rating_bracket}
                    --------------------------------------------------------------------------
                    """)
    print(df_returns_per_rating_bracket)

    # deadend4 = input('Next step...press Enter...')

    return df_returns_per_rating_bracket, df_pricedata_per_group_wo_na, df_rating_bracket, df_pricedata


def weights(num_stocks):
    """
    Produce random weights for portfolio stocks, constraining them to be additive to one
    # produce x random weight which adds up to one (where x is the number of stocks in the group)
    :param num_stocks:
    :return:
    """
    k = np.random.rand(num_stocks)
    return k / sum(k)

group_AAA_A = pricedata_per_ratingbracket()[0]
# group_Aminus = pricedata_per_ratingbracket(['A-'])[0]
# group_BBBplus = pricedata_per_ratingbracket(['BBB+'])[0]
# group_BBB = pricedata_per_ratingbracket(['BBB'])[0]
# group_BBBminus = pricedata_per_ratingbracket(['BBB-'])[0]
group_BBplus_Bminus = pricedata_per_ratingbracket(['BB+', 'BB', 'BB-', 'B+', 'B', 'B-'])[0]


def returns_and_stdevs(returns):
    """
    Calculate mean daily return and daily stdev of portfolio i

    1. Calculate the mean return for each portfolio and store the result in a matrix.

    2. Create the weight-matrix by recalling the precedent func "weights", with as many weights as the number of
    stocks in the portfolio

    3. Create the covariance-matrix, by taking the return matrix and apply the covariance function.

    4. Calculate the return of the portfolio by creating the dot-product between the weight-vector and the transposed
    mean daily return vector

    5. Calc. the standard deviation of the portfolio by taking the square root of the dot-product between the
    weight-vector, the covariance-matrix and the transposed of the weight-vector.


    :param returns: pricedata_per_ratingbracket()[0], pricedata_per_ratingbracket(['A-'])[0] etc.
    :return:
    """
    mean_returns_vector = np.asmatrix(np.mean(returns, axis=1))
    #print(mean_returns_vector)
    # shape 0 ergibt nicht die Anzahl Aktien - debuggen...
    weights_vector = np.asmatrix(weights(returns.shape[0])) # returns.shape[0] is the number of stocks per group
    #print('Returns shape aka num stocks per group: ', returns.shape[0])
    #print(weights_vector)
    covariance_matrix = np.asmatrix(np.cov(returns))
    #print(covariance_matrix)
    return_portfolio = weights_vector * mean_returns_vector.T
    #print(return_portfolio)
    stdev_portfolio = np.sqrt(weights_vector * covariance_matrix * weights_vector.T)
    #print(stdev_portfolio)
    return return_portfolio, stdev_portfolio

# Create 100,000 portfolios, and record mean daily returns and daily standard deviations for each of them

"""
For testing all steps with deadends, use just n=1
"""

# n = 100000
# für eine bessere Grafik nur 10000 nehmen
n = 10
# oben das print herausnehmen, sonst bekomme ich 100000 Zeilen mit return_port und stdev_port

"""
   numpy.column_stack
   >>> a = np.array((1,2,3))
   >>> b = np.array((2,3,4))
   >>> np.column_stack((a,b))
   array([[1, 2],
          [2, 3],
          [3, 4]])
"""

# Achtung: nicht mit der Funktion in der Funktion callen - dadurch druckt es mir jeden DF 10mal:
# um Gottes Willen - die Funktion mit einer Funktion aufrufen hat zu einer Teufelsschleife geführt;
# was normalerweise 30 sek dauert, hat 1,5h gedauert

df_returns_per_rating_bracket = pricedata_per_ratingbracket()[0]
df_returns_per_rating_bracket_Bminus = pricedata_per_ratingbracket(['BB+', 'BB', 'BB-', 'B+', 'B', 'B-'])[0]
means, stds = np.column_stack([returns_and_stdevs(df_returns_per_rating_bracket) for x in tqdm(range(n))])
means_b, stds_b = np.column_stack([returns_and_stdevs(df_returns_per_rating_bracket_Bminus) for x in tqdm(range(n))])

print(f"""
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    {type(means)}
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    {means}
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    {type(stds)}
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    {stds}
    ---------------------------------------------------------------
    ===============================================================
""")

from investing.myutils import flatten
def plot_portfolio_daily_returns_stdev():
    """
    Plot portfolios' mean daily returns and daily standard deviations previously created
    :return:


    """
    stdsf = flatten(stds_b)
    meansf = flatten(means_b)
    fig_returns = go.Figure()

    fig_returns.add_trace(go.Scatter(x=stdsf, y=meansf, mode='markers', name='markers', marker_color='steelblue'))

    fig_returns.update_layout(
        # Set the global font
        font={
            'family': 'Arial',
            'size': 16
        },
        # Update title font
        title={
            "text": f"Return and volatility of {n} portfolios for a chosen timewindow",
            "y": 0.9,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                "family": "Arial",
                "size": 20,
                "color": colors['slate-grey-darker']
            }
        }
    )
    # Add X and Y labels
    fig_returns.update_xaxes(title_text='Standard Deviation')
    fig_returns.update_yaxes(title_text='Mean of daily returns')

    fig_returns.show()

    return fig_returns


# Function that implements the same calculation explained above, with the exception that it returns
# Sharpe Ratio and weights instead of return and standard deviation

def sharpe_ratios_and_weights(varreturns):
    """
    The scope of the analysis is to take the portfolio which maximizes the Sharpe ratio.
    Therefore, among the 100,000 simulated portfolios, the portfolio with the maximum Sharpe Ratio is going to be
    selected, and then considered in the comparative analysis, both against the other groups
    (to compare portfolios with different classes of issuer credit risk) and over time
    (to see if the effect is due to a structural component of the market,
    or if Covid-19 changed the physiognomy of the latter).

    The Sharpe Ratio is calculated by dividing the portfolio return by its volatility (the standard deviation of the
    portfolio calculated for the same time frame as for the portfolio returns):

    𝑆h𝑎𝑟𝑝e𝑅𝑎𝑡𝑖𝑜 𝑇 = 𝑃𝑜𝑟𝑡𝑓𝑜𝑙𝑖𝑜𝑅𝑒𝑡𝑢𝑟𝑛 𝑇 / 𝑃𝑜𝑟𝑡𝑓𝑜𝑙𝑖𝑜 𝑆𝑡𝑎𝑛𝑑𝑎𝑟𝑑 𝐷𝑒𝑣𝑖𝑎𝑡𝑖𝑜𝑛 𝑇
    :param varreturns:
    :return:
    """
    mean_returns_vector = np.asmatrix(np.mean(varreturns, axis=1))
    weights_vector = np.asmatrix(
        weights(varreturns.shape[0]))  # ich nehme nullte Stelle aus der x-y-Form der Matrix, hier also die rows
    '''
    >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4],
    ...                    'col3': [5, 6]})
    >>> df.shape
    (2, 3) #2 Reihen, 3 Spalten
    '''
    covariance_matrix = np.asmatrix(np.cov(varreturns))
    return_portfolio = weights_vector * mean_returns_vector.T  # NumPy: Transpose ndarray (swap rows and columns, rearrange axes)
    stdev_portfolio = np.sqrt(weights_vector * covariance_matrix * weights_vector.T)
    sharpe_ratio = float(return_portfolio / stdev_portfolio)
    return sharpe_ratio, weights_vector


# Simulate 100,000 portfolios and store the Sharpe Ratios and weights for each of them
# n = 100000
# n = 10

if __name__ == '__main__':
    # get_current_tickerlist_sp500()

    #group_AAA_A = pricedata_per_ratingbracket()[1:2]    # oder [0:2]???

    #group_AAA_A_pricedata = pricedata_per_ratingbracket()[0]

    #group_AAA_A_stdev = returns_and_stdevs(group_AAA_A_pricedata)

    #tickers_as_categories_plotted()
    #plot_portfolio_daily_returns_stdev()
    None





def tickers_as_categories_plotted():
    """
    The matplotlib-statement is just for comparison reasons - in production we use Plotly
    # Convert credit ratings into categorical data and order them
    :return:
    """

    # df_available_stocks = data_collector['df_available_stocks']
    df_tickers_rating = data_collector['df_tickers_rating']

    df_tickers_rating.set_index('Symbol', inplace=True)

    # print(df_tickers_rating)

    to_plot = df_tickers_rating['LT-Rating_orig'].value_counts() \
        [['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-']]
    # print(to_plot)

    # Create bar graph
    ax = to_plot.plot.bar(figsize=(8, 6))

    # Insert bar labels
    for i in ax.patches:
        ax.text(i.get_x() + 0.25, i.get_height() + 1.5, str(round((i.get_height()))), fontsize=10, color='dimgrey',
                ha='center')

    # plt.box(False)
    # plt.show()

    """
    With Plotly:

    Legacy - with plotly express:
    graph_data = go.Bar(
        y=df_tickers_rating['LT-Rating_orig'].value_counts() \
        [['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-']],
        hoverinfo='all')

    #figure = go.Figure(data=[graph_data])

    figure = px.bar(df_tickers_rating['LT-Rating_orig'].value_counts()\
            [['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-']],
                    hover_data=[], color=df_tickers_rating['LT-Rating_orig'].value_counts()
                    )
    """

    # Todo: let's use graph objects go for that we can be more flexible in the design:

    figure = go.Figure(data=[go.Bar(
        x=['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-'],
        y=df_tickers_rating['LT-Rating_orig'].value_counts()[
            ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-']],
        marker=go.bar.Marker(autocolorscale=True, color='orange'),
        hovertemplate='<br>'.join(
            ['Rating class: %{x}', 'No. of constituents: %{y}']
        ))])

    figure.update_layout(
        # Set the global font
        font={
            'family': 'Raleway',  # Overpass
            'size': 16
        },
        # Update title font
        title={
            "text": "Distribution of S&P-500-Ratings",
            "y": 0.9,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                "family": "Raleway",  # Overpass
                "size": 20,
                "color": colors['slate-grey-darker']
            }
        }
    )

    # Add X and Y labels
    figure.update_xaxes(title_text='LT-Rating Groups')
    figure.update_yaxes(title_text='Count')

    figure.show()







