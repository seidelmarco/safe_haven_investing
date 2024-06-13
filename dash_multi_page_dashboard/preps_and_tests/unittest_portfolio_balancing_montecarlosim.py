"""
Our ultimate goal: create portfolio weights that will maximize the Sharpe Ratio of the portfolio as a whole.

Carrying out a Monte Carlo simulation that will generate weights based on the highest Sharpe Ratio.
Second: rather than playing a guessing game, we can use SciPy (Python library) in order to determine
what these optimal weights would be.

https://www.investopedia.com/articles/financial-theory/11/calculating-covariance.asp
"""

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

from investing.myutils import flatten

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

from scipy.optimize import minimize


settings = {
    'max_columns': None,
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

    """
        Cleaning and cleansing:
        wir wollen alle, die kein Rating haben, löschen
        it could be that not all sp-constituents have a rating thus we need to construct a dataframe with
        available stocks:
        """

    df = df_available_stocks.copy()
    df.drop(['level_0', 'index'], axis=1, inplace=True)
    df.dropna(axis=0, subset=['LT-Rating_orig'], inplace=True)

    # securities with rating as tickers-list:
    tickers_rating_as_list = df['Symbol'].tolist()

    df_complete_all_ratings = df

    # Ticker als DF:
    df_tickers_rating = df_complete_all_ratings[['Symbol', 'LT-Rating_orig', 'security', 'sectors', 'subindustries']]

    """
    #################################################
    
    We wish to aggregate sectors and subindustries and construct filter-lists on the fly:
    
    #################################################
    """

    df_all_sectors = df_available_stocks.copy()
    df_all_sectors.drop(['level_0', 'index'], axis=1, inplace=True)
    df_all_sectors = df_all_sectors[['Symbol', 'security', 'sectors', 'subindustries']]

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

    print('.filter - scheint nicht zu funkt. ', df_sectors_subindustries_grouped.filter(like='Utili', axis=0))

    # Alternative: isin; weil wir ja nur für die Grafiken filtern wollen
    df_sectors_grouped_isin = None


    # Aggregate-function:
    df_sectors_agg = df_all_sectors.groupby(['sectors', 'subindustries'], observed=True).sectors.agg(['count'])

    df_sectors_agg2 = df_all_sectors.groupby(['sectors', 'subindustries'], observed=True).aggregate({'sectors': 'count', 'subindustries': 'count'})

    # Pivot-Table:
    df_pivot = df_available_stocks.drop(['level_0', 'index', 'Symbol', 'url', 'headquarters', 'Creditwatch', 'Remarks'], axis=1)
    df_pivot = df_pivot.set_index(['security'])
    df_pivot = df_pivot.astype({
        'sectors': 'category',
        'subindustries': 'category',
        'Sectors': 'category',
        'Subindustries': 'category'
    })
    # print(df_pivot)
    print('df_pivot-Datatypes: \n', df_pivot.dtypes)
    # Generally, categorical columns are used as indexes.
    # df_sectors_agg_pivot = pd.pivot_table(data=df_pivot, values=[], index=['security'])
    #print('Pivot Table: \n', df_sectors_agg_pivot)
    # TypeError: agg function failed [how->mean,dtype->object]

    # Multiindex: index=['sectors', 'subindustries'], values=['daily_return']
    """
    https://www.analyticsvidhya.com/blog/2020/03/pivot-table-pandas-python/
    
    heatmap_results = pd.pivot_table(results_agg, values='Price',
                                 index=['Destination'],
                                 columns='Start')

    # TypeError: Image data of dtype object cannot be converted to float
    # someone figured it out after a while: Appears that the type of DF.values was set to object.
    # It can be fixed by the following line:
    # Wenn wir die Preise als Annotationen in jeder Zelle der Heatmap anzeigen wollen, müssen wir diese Column
    # nochmal in float oder integer umwandeln:
    
    heatmap_results = heatmap_results[heatmap_results.columns].astype(float)  # or int
    
    aggfunc is an aggregate function that pivot_table applies to your grouped data.

    By default, it is np.mean(), but you can use different aggregate functions for different features too! 
    Just provide a dictionary as an input to the aggfunc parameter with the feature name as the key and the 
    corresponding aggregate function as the value.
    
    But what are you aggregating on? You can tell Pandas the feature(s) to apply the aggregate function on in the 
    value parameter. The value parameter is where you tell the function which features to aggregate on. 
    It is an optional field, and if you don’t specify this value, then the function will aggregate all the numerical 
    features of the dataset:
    
    Using multiple features as indexes is fine, but using some features as columns will help you to intuitively 
    understand the relationship between them. Also, the resultant table can always be better viewed by incorporating 
    the columns parameter of the pivot_table.

    The columns parameter is optional and displays the values horizontally on the top of the resultant table. 
    Both columns and the index parameters are optional, but using them effectively will help you to intuitively 
    understand the relationship between the features.
    
    Fill nans with the avg: fill_value=np.mean(df['Age'])
    """

    print(f"""
    
    ##########################################################
            Dataframe with all sectors, rating wont matter:
            Length: {df_all_sectors.shape[0]}
            
            {df_all_sectors.describe(include='all').T}
            
            {df_all_sectors}
            {df_all_sectors.dtypes}
            
            We group by sectors and show one result:
            {df_sectors_grouped}
            
            Aggregate-function:
            {df_sectors_agg}
            
            Alternative Aggregate-Funktion:
            {df_sectors_agg2}
        
    ##########################################################
        """)

    """
    #####################################################
        Download daily stock price data for S&P500 stocks
        rating_bracket: list with strings:
    #####################################################
    """

    #df_rating_bracket = df_tickers_rating[df_tickers_rating['LT-Rating_orig'].isin(rating_bracket)] \
        #.sort_values(by='LT-Rating_orig', ascending=True)

    # Test 3 Ticker als DF:
    rating_bracket_test3n = ['AAPL', 'MSFT', 'PCAR', 'TRGP', 'TDG', 'TSLA', 'AMAT', 'DE']

    df_rating_bracket = df_tickers_rating[df_tickers_rating['Symbol'].isin(rating_bracket_test3n)] \
        .sort_values(by='Symbol', ascending=True)

    # Test 2 Ticker als DF:
    rating_bracket_test2n = ['PCAR', 'TSLA']
    #df_rating_bracket = df_tickers_rating[df_tickers_rating['Symbol'].isin(rating_bracket_test2n)] \
        #.sort_values(by='Symbol', ascending=True)

    df_sectors_unique = df_rating_bracket['sectors'].unique()

    df_subindustries_unique = df_rating_bracket['subindustries'].unique()

    print('Rating-bracket-shape rows, cols: ', df_rating_bracket.shape)
    ser_rating_bracket = df_rating_bracket['Symbol']
    print(f""" 
        --------------------------------------------------------------------------
        Dataframe Tickers/Securities with ratings:
        you see {ser_rating_bracket.size} Elements for the bracket {rating_bracket}
        --------------------------------------------------------------------------
                """)
    print(df_rating_bracket)
    print('\n')
    print(df_sectors_unique)
    print('\n')
    print(df_subindustries_unique)

    """
        Download daily stock price data per S&P500-group
    """

    # die Symbols-List beinhaltet noch JBL, was wir wegen Nans löschen - könnte später ein Problem werden
    symbols_list = ser_rating_bracket.to_list()

    # retrieve the whole df pricedata - dates_as_index=True um einen sortierbaren Index zu haben
    df_pricedata_complete = pull_df_from_db_dates_or_not(sql='sp500_adjclose', dates_as_index=True)

    # Multiindex Date and ID:

    df_pricedata_complete = df_pricedata_complete.reset_index().set_index(['Date', 'ID'])

    # slice just the needed columns per rating_bracket with the var symbols_list:
    df_pricedata_per_rating_bracket = df_pricedata_complete[symbols_list].sort_index(ascending=True)

    # we need a sectors list: use agg-func - use pivot-table
    # df pricedata per sector = df_pricedata_complete[sectors_list].sort_index(ascending=True)

    """
        query eine Datumsklammer
    """
    # startdate = "2023-10-20"
    # enddate = "2024-03-04"
    timedelta = dt.datetime.now() - dt.timedelta(200)
    startdate = timedelta.strftime('%Y-%m-%d')
    enddate = dt.datetime.now().strftime('%Y-%m-%d')
    df_pricedata_per_rating_bracket.query('Date >= @startdate & Date <= @enddate', inplace=True)

    """
    Drop nans:
    """

    #print(f"""
        #-------------------------------------------------------------
            #You see the df pricedata per rating bracket WITH nans:
            #JBL hat erst wenige Tage mit Daten - aktuell löschen wir die Spalten
            #in deren Zeitklammer nans stecken...
            #{df_pricedata_per_rating_bracket.shape}
            #{df_pricedata_per_rating_bracket}
        #-------------------------------------------------------------
        #""")

    df_pricedata_per_group_wo_na = df_pricedata_per_rating_bracket.dropna(axis=1)
    print(f"""
    -------------------------------------------------------------
        You see the df pricedata per rating bracket w/o nans:
        
        {df_pricedata_per_group_wo_na.shape}
        {df_pricedata_per_group_wo_na}
    -------------------------------------------------------------
    """)

    # An dieser Stelle müssen wir eine neue symbols_list ohne nans schaffen:
    symbols_list_no_nans = df_pricedata_per_group_wo_na.columns.values.tolist()
    print(len(symbols_list_no_nans), symbols_list_no_nans)


    """
    Now that we’ve collected the data, we need to understand our ultimate goal: create portfolio weights that will 
    maximize the Sharpe Ratio of the portfolio as a whole. However, we must discuss what the Sharpe Ratio 
    is first before we attempt to manipulate it in any way.

    The Sharpe Ratio is defined as the difference between return and the risk-free rate (which we usually assume 
    to be 0), divided by the volatility, which we usually use standard deviation for.
    Sharpe Ratio Formula

    The next thing we need to do is generate weights randomly for each stock (we divide by the total sum of the 
    weights in order to ensure that the weights add up to 1). Next, we retrieve the expected portfolio return 
    and volatility for the corresponding stocks.

    We can then use this information to calculate the Sharpe Ratio for one instance of weight generations. 
    In order to calculate the maximum Sharpe ratio for a set of weights, we need to repeat this process thousands 
    of times in a Monte Carlo simulation.
    """

    """
    Expected return:
    Transform the price matrix into a return matrix:
    𝑅𝑒𝑡𝑢𝑟𝑛𝑖,𝑇 = 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇 − 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1/𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 = 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇/𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 − 1 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1
    shift(1) geht eine Zeile zurück (schiebt eine Zeile von oben auf den cursor (die aktuelle loop))
    """

    # 1. Mean of daily returns - period=1 daily:
    mean_daily_ret = df_pricedata_per_group_wo_na.pct_change(1).mean()
    print(f"""
        1. Mean of daily returns - period=1 daily:
        
        If the mean of all daily returns in a given time period is below zero, it shows that
        the price on the last day of our time bracket - the alleged day of selling - is still
        below our startingpoint - the alleged day of buying.
        That's exactly the info we need to avoid losses:
        
        WE NEED THE AVERAGE RETURN AND THE DAILY RETURN OF EACH STOCK-PAIR FOR THE COV-FORMULA :-)
        
        {mean_daily_ret}
        """)

    # 2. Pricedata pct_change period 1 correlation:
    corr_daily_ret = df_pricedata_per_group_wo_na.pct_change(1).corr()

    # find the top correlations:
    # corr_daily_ret_top10 = corr_daily_ret.max().max()

    corr_arr = corr_daily_ret.unstack()
    sorted_corr_arr = corr_arr.sort_values(kind="quicksort", ascending=False).drop_duplicates()
    print(f"""
    
         Uses of Covariance

        Covariance can tell how the stocks move together, but to determine the strength of the relationship, 
        look at their correlation. The correlation should, therefore, be used in conjunction with the covariance, 
        and is represented by this equation:

        Correlation=ρ=cov(X,Y)/σXσY
        where:cov(X,Y)=Covariance between X and Y
        σX=Standard deviation of X
        σY=Standard deviation of Y
        Matrix of all correlations:
        
        {corr_daily_ret}, 
        
        DF-values (yields the same result as df.to_numpy :-) <class 'numpy.ndarray'>)
        {corr_daily_ret.values}, 
        {sorted_corr_arr}
        
        """)

    # find the highest correlations - drop self correlations:
    sorted_corr_arrtop10 = sorted_corr_arr.nlargest(10)

    # quantile_percent_upper: float = 0.99,
    #print(
        #data_no_mv.sort_values(by=['Price'], ascending=False).head(int((data_no_mv['Brand'].count() * 0.05).round(0))))
    print(f"""
        2. Pricedata pct_change period 1 daily correlation:
            - hier später weiterarbeiten und die Korrelationskoeffzienten nach den TOP 10 filtern,
            um ein passendes Aktienportfolio zu bilden:
            
        We see the correlation-coefficient which means the covariance divided by the product of the var-stdevs;
        so we obtain a normalized comparable value/ratio
        
        It is calculated by summing -up all covariances of all datapoints (days)
        
        We obtain the dependence of the variance of one stock from the variance of another stock.
        
        So we should look for a postitive benchmark-stock and then for stocks which follow in the same direction.
        
        You see high corr e. g. for XOM-CVX or homebuilders like LEN and PHM
        This means, we can work around this observation by a sector-clustering...TODO
        
        Correlations of daily returns Top10:
        {sorted_corr_arr}
        {sorted_corr_arrtop10}
    """)

    # Returns stimmen - mit XOM geprüft:
    df_returns_per_rating_bracket = (df_pricedata_per_group_wo_na / df_pricedata_per_group_wo_na.shift(1) - 1)[1:]

    print(f"""
        3. Expected Portfolio Return
        --------------------------------------------------------------------------
        You see {len(symbols_list_no_nans)} daily returns for the bracket {rating_bracket}
        --------------------------------------------------------------------------
        """)
    print(df_returns_per_rating_bracket)

    df_pricedata_per_group_wo_na_no_id = df_pricedata_per_group_wo_na.reset_index().drop(['ID'], axis=1).set_index('Date')
    pricedata_normed = df_pricedata_per_group_wo_na_no_id / df_pricedata_per_group_wo_na_no_id.iloc[0]

    # nur eine Zeile für linechart mit matplotlib :-)
    #pricedata_normed.plot(figsize=(16, 8))
    #plt.show()

    # 3. Pricedata daily returns period=1 daily:
    pricedata_daily_ret = df_pricedata_per_group_wo_na.pct_change(1)
    print('3. Pricedata pct_change period 1 daily returns:')
    print(pricedata_daily_ret)

    pricedata_daily_ret_log = np.log(df_pricedata_per_group_wo_na/df_pricedata_per_group_wo_na.shift(1))
    print(f"""
        Zum Vergleich mit meinen Returns - sollte exakt das gleiche Ergebnis sein :-):
        {df_returns_per_rating_bracket}
        
        Als dritten Vergleich die log-normalisierten returns; ergeben eine andere Rundung:
        {pricedata_daily_ret_log}
        
        Log-returns described:
        {pricedata_daily_ret_log.describe().transpose()}
    """)

    pricedata_daily_ret_log.hist(bins=20, figsize=(12, 6))
    plt.tight_layout()
    #plt.show()

    # 4. Calculate mean of daily returns and multiply by days of observed timeframe:
    mean_daily_ret_log = pricedata_daily_ret_log.mean()
    print(f"""
        4. Calculate mean of daily returns and multiply by days of observed timeframe:
        PS: the mean should be the same result like in the describe-func :-)
        Return of the mean of all daily returns:
        {mean_daily_ret_log}
        Return of the timeframe:
        
        Todo: show the timeframe-return as hoverdata in the plot var: mean_daily_ret_log_timeframe
        {mean_daily_ret_log*len(pricedata_daily_ret_log)}
    """)

    # 4.1 Calculate Median Win or Loss of the time period:
    # first_last_diff = df_pricedata_per_group_wo_na.iloc[-1, :] - df_pricedata_per_group_wo_na.iloc[0, :]
    first_last = df_pricedata_per_group_wo_na.iloc[[0, -1]]
    first_last_diff = first_last.pct_change()
    print(f"""
        Win/loss in timeframe:
        {first_last}
        {first_last_diff}
    """)

    # 5. Compute pairwise covariance of columns
    pricedata_daily_cov = pricedata_daily_ret_log.cov()
    print(f"""
            5. Compute pairwise covariance of columns:
            
            You see the variance-covariance-matrix - the diagonal like PCAR-PCAR shows the variance of 1 stock
            all other cells show the covariance of two stocks:
            
            {pricedata_daily_cov}
            
            Multiplied by timeframe:
            {pricedata_daily_cov*len(pricedata_daily_cov)}
        """)

    """ 
    -----------------------------------
    6. Single run for generating values
    -----------------------------------
    """

    # Set seed (optional) - only for debugging to obtain the same set of weights everytime:
    # np.random.seed(101)

    # Create Random Weights - divided by sum of weights to ensure weights add to 1.0
    """
    The next thing we need to do is generate weights randomly for each stock (we divide by the total sum 
    of the weights in order to ensure that the weights add up to 1). 
    Next, we retrieve the expected portfolio return and volatility for the corresponding stocks.

    We can then use this information to calculate the Sharpe Ratio for one instance of weight generations. 
    In order to calculate the maximum Sharpe ratio for a set of weights, we need to repeat this process 
    thousands of times in a Monte Carlo simulation.
    """
    def weights_func(num_stocks):
        """

        :param num_stocks:
        :return:
        """
        # print('Creating  Weights:')
        weights = np.array(np.random.rand(len(num_stocks)))  # num stocks
        weights_to_1 = weights / np.sum(weights)
        # print(weights_to_1)
        # print('\n')
        return weights_to_1

    weights = weights_func(symbols_list_no_nans)
    days = len(pricedata_daily_ret_log)

    print(f"""
            The next thing we need to do is generate weights randomly for each stock (we divide by the total sum 
            of the weights in order to ensure that the weights add up to 1). 
            Next, we retrieve the expected portfolio return and volatility for the corresponding stocks.
        
            We can then use this information to calculate the Sharpe Ratio for one instance of weight generations. 
            In order to calculate the maximum Sharpe ratio for a set of weights, we need to repeat this process 
            thousands of times in a Monte Carlo simulation.
          """)

    # Expected return
    # we don't need a dot-product here, since its just one-dimensional:
    exp_ret = np.sum(pricedata_daily_ret_log.mean() * days * weights)

    # Expected Variance
    """
    Reasoning: 
    Your result will be a scalar if you use vectors since the dot-product 
    is just the sum of the products of its corresponding elements
    Matrixes:
    Your result will be a m x k-matrix if you multiply m x n and n x k...!
    """

    # Look at 5. for the reasoning:
    exp_volatility = np.sqrt(np.dot(weights.T, np.dot(pricedata_daily_ret_log.cov() * days, weights)))

    # Sharpe Ratio
    sr = exp_ret/exp_volatility

    print(f"""
                6. Compute expected portfolio return for one random example/one instance of weight generation:
                Anzahl Gewichte: {len(weights)}
                days = {days}
                Create Random Weights - divided by sum of weights to ensure weights add to 1.0
                Weights: {weights}
                
                Expected portfolio return:
                exp_ret = np.sum(pricedata_daily_ret_log.mean() * days * weights)
                (Reasoning: the whole information of all days already lies in the mean() - that's why we can
                multiply by the days)
                {exp_ret}
                
                Expected Variance/volatility:
                exp_volatility = np.sqrt(np.dot(weights.T, np.dot(pricedata_daily_ret_log.cov() * days, weights)))
                
                (Reasoning: the whole information of all days already lies in the cov() - that's why we can
                multiply by the days)
                {exp_volatility}
                
                Sharpe Ratio:
                {sr}

                
            """)
    # --------------------------------------
    # Monte-Carlo-Simulation:
    # --------------------------------------
    num_ports = n_portfolios

    all_weights = np.zeros((num_ports, len(symbols_list_no_nans)))  # Shape is a tuple and yields a matrix num_ports * stocks
    print(f"""
        Number and values of all weights:
        We create a matrix filled with Zeros in {len(all_weights)} rows (n of repetitions) 
        {len(symbols_list_no_nans)} columns (number of stocks)
        
        Number: {len(all_weights)}
        
        Values: {all_weights}
        """)

    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    print(f"""Prepared empty 1D-arrays with Zeros of length of the num-portfolios:
            Return-array:
            {ret_arr} 
            Volatility-array:
            {vol_arr} 
            SR-array:
            {sharpe_arr}
            """)

    for i in range(num_ports):
        # Create Random Weights

        weights_arr = np.array(np.random.rand(len(symbols_list_no_nans)))

        # Rebalance Weights
        weights_arr = weights_arr / np.sum(weights_arr)
        # print('Weights per portfolio: ', weights_arr)

        # die func von oben scheint nicht zu callen sein:
        # TypeError: 'numpy.ndarray' object is not callable, vermutlich, weil es in einer Schleife ist
        # print('Weights per portfolio: ', weights_func(len(symbols_list_no_nans)))

        # Save weights - geilo, dass ist die übelste Abkürzung für append oder list-comprehension :-):
        all_weights[i, :] = weights_arr

        # Expected return:
        ret_arr[i] = np.sum(pricedata_daily_ret_log.mean() * weights_arr * days)
        # print(ret_arr[i])

        # Expected Variance
        vol_arr[i] = np.sqrt(np.dot(weights_arr.T, np.dot(pricedata_daily_ret_log.cov() * days, weights_arr)))
        # print(vol_arr[i])

        # Sharpe Ratio
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
        # print(sharpe_arr[i])

        pos_weights_maxsr = all_weights[sharpe_arr.argmax(), :]
        all_weights_unstacked = pos_weights_maxsr.tolist()

        # rating_bracket_test3n oder df_rating_bracket
        # portfolio_best = dict(zip(pos_weights_maxsr, all_weights_unstacked))
        portfolio_best = dict(zip(rating_bracket_test3n, all_weights_unstacked))

    print(f"""
            MONTE-CARLO-SIMULATION
            ---------------------------------------------
            All weights from the for-loop for {num_ports} portfolios

            {all_weights}
            
            all_weights_unstacked:
            {all_weights_unstacked}

            ----------------------------------------------""")

    max_sr = sharpe_arr.max()
    max_ret = ret_arr.max()
    print(f"""
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
    
    {portfolio_best}
    
    -----------------------------------""")

    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
    print('Return at max. Sharpe Ratio: ', max_sr_ret, 'Volatility of max. Sharpe-Ratio: ', max_sr_vol)

    return df_returns_per_rating_bracket, pricedata_daily_ret_log, days, symbols_list_no_nans, \
            vol_arr, ret_arr, sharpe_arr, max_sr_vol, max_sr_ret, pricedata_normed, max_sr, df_all_sectors, \
            df_sectors_grouped, df_sectors_subindustries_grouped, df_sectors_agg

# -------------------------------
# For testing purposes:
# -------------------------------

df_returns_per_rating_bracket, pricedata_daily_ret_log, days, symbols_list_no_nans, vol_arr, ret_arr, sharpe_arr, \
    max_sr_vol, max_sr_ret, pricedata_normed, max_sr, df_all_sectors, df_sectors_grouped, \
    df_sectors_subindustries_grouped, df_sectors_agg = main_calculations(n_portfolios=10000)

# ------------------------------------------------------------------------------------------------------

df_returns_per_rating_bracket_A, pricedata_daily_ret_log_A, days_A, symbols_list_no_nans_A, vol_arr_A, ret_arr_A, sharpe_arr_A, \
    max_sr_vol_A, max_sr_ret_A, pricedata_normed_A, max_sr_A, df_all_sectors_A, df_sectors_grouped_A, \
    df_sectors_subindustries_grouped_A, df_sectors_agg_A = main_calculations(n_portfolios=10000)

df_returns_per_rating_bracket_Amin, pricedata_daily_ret_log_Amin, days_Amin, symbols_list_no_nans_Amin, vol_arr_Amin, ret_arr_Amin, sharpe_arr_Amin, \
    max_sr_vol_Amin, max_sr_ret_Amin, pricedata_normed_Amin, max_sr_Amin, df_all_sectors_Amin, df_sectors_grouped_Amin, \
    df_sectors_subindustries_grouped_Amin ,df_sectors_agg_Amin = main_calculations(['A-'], n_portfolios=10000)    # ['BBB-'],

df_returns_per_rating_bracket_3Bplus, pricedata_daily_ret_log_3Bplus, days_3Bplus, symbols_list_no_nans_3Bplus, vol_arr_3Bplus, ret_arr_3Bplus, sharpe_arr_3Bplus, \
    max_sr_vol_3Bplus, max_sr_ret_3Bplus, pricedata_normed_3Bplus, max_sr_3Bplus, df_all_sectors_3Bplus, df_sectors_grouped_3Bplus, \
    df_sectors_subindustries_grouped_3Bplus, df_sectors_agg_3Bplus = main_calculations(['BBB+'], n_portfolios=10000)    # ['BBB-'],

df_returns_per_rating_bracket_3B, pricedata_daily_ret_log_3B, days_3B, symbols_list_no_nans_3B, vol_arr_3B, ret_arr_3B, sharpe_arr_3B, \
    max_sr_vol_3B, max_sr_ret_3B, pricedata_normed_3B, max_sr_3B, df_all_sectors_3B, df_sectors_grouped_3B, \
    df_sectors_subindustries_grouped_3B, df_sectors_agg_3B = main_calculations(['BBB'], n_portfolios=10000)    # ['BBB-'],

df_returns_per_rating_bracket_3Bmin, pricedata_daily_ret_log_3Bmin, days_3Bmin, symbols_list_no_nans_3Bmin, vol_arr_3Bmin, ret_arr_3Bmin, sharpe_arr_3Bmin, \
    max_sr_vol_3Bmin, max_sr_ret_3Bmin, pricedata_normed_3Bmin, max_sr_3Bmin, df_all_sectors_3Bmin, df_sectors_grouped_3Bmin, \
    df_sectors_subindustries_grouped_3Bmin, df_sectors_agg_3Bmin = main_calculations(['BBB-'], n_portfolios=10000)    # ['BBB-'],

df_returns_per_rating_bracket_Bmin, pricedata_daily_ret_log_Bmin, days_Bmin, symbols_list_no_nans_Bmin, vol_arr_Bmin, ret_arr_Bmin, sharpe_arr_Bmin, \
    max_sr_vol_Bmin, max_sr_ret_Bmin, pricedata_normed_Bmin, max_sr_Bmin, df_all_sectors_Bmin, df_sectors_grouped_Bmin, \
    df_sectors_subindustries_grouped_Bmin, df_sectors_agg_Bmin = main_calculations(['BB+', 'BB', 'BB-', 'B+', 'B', 'B-'], n_portfolios=10000)    # ['BBB-'],

d = {'Ind. Sel.': max_sr, 'AAA - A': max_sr_A, 'A-': max_sr_Amin, 'BBB+': max_sr_3Bplus, 'BBB': max_sr_3B, 'BBB-':max_sr_3Bmin, 'B-': max_sr_Bmin }
df_all_max_sr = pd.DataFrame(data=d, index=[0])
print(df_all_max_sr)


def get_ret_vol_sr(weights):
    """
    Carrying out a Monte Carlo simulation along with a SciPy minimization function to maximize the overall Sharpe Ratio
    of a certain stock portfolio.
    We can also use SciPy in order to mathematically minimize the negative sharpe ratio,
    giving it's maximum possible SR

    Takes in weights, returns array or return,volatility, sharpe ratio
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

# 0-1 bounds for each weight
# bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
# print(bounds)
# print (len(bounds))
bounds = [(0, 1) for i in range(len(symbols_list_no_nans))]
print(bounds)
print(len(bounds))

# Initial Guess (equal distribution)
#init_guess = [0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0555,0.0565]
#print(len(init_guess))
init_guess = [0.0555 for i in range(len(symbols_list_no_nans) - 1)] + [0.0565]
print(len(init_guess))

# Sequential Least SQuares Programming (SLSQP)
opt_results = minimize(fun=neg_sharpe, x0=init_guess, method='SLSQP', bounds=bounds, constraints=cons)

print(f"""
    Optimal results from scipy-minimize:
    {opt_results}
    
    Optimal results .x (best Return, volatility and SR) passed to get_ret_vol_sr-func scipy-minimize:
    {get_ret_vol_sr(opt_results.x)}""")

# Our returns go from 0 to somewhere along 0.3
# Create a linspace number of points to calculate x on
# the frontiers come from our first figure - choose it manually:

# linear spaced ndarray:
# https://realpython.com/np-linspace-numpy/
# vector-space a.k.a. linear-space - num=50 is default:
# list(range(1, 11) is limited to integers, so we cannot use it
# You can still use range() with list comprehensions to create non-integer ranges - look at file basics_panda_fundamentals.py...
frontier_y = np.linspace(start=0.10, stop=0.30, num=101) # Change 101 to a lower number for slower computers!


def minimize_volatility(weights):
    """

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

"""
# Only for rapid prototyping:
plt.figure(figsize=(12, 8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')

# Add red dot for max Sharpe Ratio
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')

# Add frontier line
plt.plot(frontier_volatility, frontier_y, 'b--', linewidth=3)
plt.show()
"""


"""
----------------------------------
Plotting:
----------------------------------
"""


def scatter_plot_ret_vol_sr():
    """

    :return:
    """
    figure_scat = go.Figure(data=go.Scatter(x=vol_arr, y=ret_arr, mode='markers',
                                       marker=dict(
                                           size=12,
                                           color=sharpe_arr,
                                           colorscale='Viridis',
                                           colorbar=dict(title='Sharpe<br>Ratio'),
                                           showscale=True,
                                           line_width=1,
                                           line_color='black'
                                       ),
                                       #name='Sharpe Ratio',
                                       showlegend=False,
                                       hovertext=[i for i in sharpe_arr],
                                       hoverinfo='text',))
                                       #hovertemplate=(f'<b>Risk: %{vol_arr[0]}% <br>Return: %{ret_arr[0]}%</b><BR><BR>weights:<BR>' +
                               #f'%{sharpe_arr[0]}%' +
                               #'<extra></extra>')))
    # figure.add_trace(go.Scatter(x=vol_arr, y=ret_arr, mode='markers', name='markers', marker_color='steelblue'))

    # Add frontier line
    """
    Efficient frontier connects those portfolios which offer the highest expected return for a
    specified level of risk. Portfolios below that line are considered to be sub-optimal.
    """
    figure_scat.add_trace(go.Scatter(x=frontier_volatility, y=frontier_y, mode='lines+markers', name='',
                                marker_color='darkorange'))

    # Add red dot for max Sharpe Ratio
    figure_scat.add_trace(go.Scatter(x=[max_sr_vol], y=[max_sr_ret],
                                mode='markers',
                                marker=dict(
                                    size=16,
                                    color='darkred',
                                    line_color='black',
                                    line_width=.5
                                )))

    figure_scat.update_layout(
        font={
                'family': 'Rockwell',
                'size': 20
            },
            # Update title font
        title={
                "text": f"Return and volatility of {len(symbols_list_no_nans)}-stock portfolio for {days} days",
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
    figure_scat.show()
    return figure_scat


def lineplot_daily_returns_pctchange():
    """
    Pricedata-Return starts at 1 - to see increases and decreases like in % we got to subtract 1
    :return:
    """
    df = pricedata_normed - 1
    figure = go.Figure()
    for col in df.columns:
        figure.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
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
    figure.show()
    return figure


def bar_plot_sr_all_brackets():
    """

    :return:
    """
    colors = ['dodgerblue', ] * 7
    i = df_all_max_sr.values[0].argmax()
    colors[i] = 'firebrick'
    figure_sr = go.Figure(data=[go.Bar(x=df_all_max_sr.columns, y=df_all_max_sr.values[0],
                                    marker_color=colors,
                                    ),
                             ])
    # figure.add_bar(x=df_all_max_sr.columns, y=df_all_max_sr.values[0])
    figure_sr.update_layout(bargap=.40)
    figure_sr.update_xaxes(title='Long Term Issuer Credit Rating')
    figure_sr.update_yaxes(title='Max Sharpe Ratio')

    figure_sr.show()
    return figure_sr


def barplot_all_sectors_subindustries():
    """
    df_all_sectors
    :return:
    """
    colors = ['indianred', ] * len(df_sectors_grouped.index)

    figure_sec = go.Figure(data=[go.Bar(x=df_sectors_grouped.index, y=df_sectors_grouped.values, marker_color=colors)])
    # figure.add_trace(go.Bar(x=df_sectors_agg.index, y=df_sectors_agg.values, name='Subindustries'))
    figure_sec.update_layout()
    figure_sec.update_xaxes(title='Fig.4: All sectors')
    figure_sec.update_yaxes()

    df_sectors_subindustries_grouped = df_all_sectors.groupby(['subindustries'], observed=True,
                                                              as_index=False).count()

    colors_subs = ['lightsalmon', ] * len(df_sectors_subindustries_grouped.index)

    figure2 = go.Figure(data=[go.Bar(x=df_sectors_subindustries_grouped['subindustries'].values, y=df_sectors_subindustries_grouped.Symbol, marker_color=colors_subs)])
    figure2.update_layout()
    figure2.update_xaxes(title='Fig. 5: All Subindustries')
    figure2.update_yaxes()


    """
    
    
    Hier mit den Aktien je Subindustry per Sector beginnen: in Dash mit Auswahlmöglichkeit Sektor und Subindustry:
    
    
    """

    #sector = input('Sector: ')
    #subind = input('Subindustry: ')



    # isin, query(), filter(), ==

    # ==
    df_single_sector = df_sectors_grouped.index == 'Energy'
    print(df_single_sector)

    # isin, query(), filter(), ==

    # ==
    # replace == in Dash with Dropdown:
    df_single_sector_energy = df_all_sectors[df_all_sectors['sectors'] == 'Energy']
    print(f"""
                DF_all_sectors - cleaned and cleansed source:
                {df_all_sectors}

                df_sectors_grouped - goes with the first figure (11 counted sectors):

                {df_sectors_grouped}

                df_single_sector_energy:

                {df_single_sector_energy}
                """)


    """
    df_single_sector_subindustries_grouped goes with Figure 3:
    """

    df_single_sector_subindustries_grouped = df_single_sector_energy.groupby(['subindustries'], observed=True,
                                                                             as_index=False).count()
    print(f"""

            df_single_sector_subindustries_grouped goes with Figure 3:
            {df_single_sector_subindustries_grouped}
        """)

    """
    The next df is like df_single_sector_energy and can be used as basis for Figure 3 with other sectors:
    """
    df_single_sector_symbols = df_all_sectors[df_all_sectors['sectors'] == 'Energy']

    print(f"""

                df_single_sector_symbols can  go with Figure 3:
                {df_single_sector_symbols}

                For testing - we build a list of all sector symbols:

                {[i for i in df_single_sector_symbols['security']]}


                Next step: we need a list of list of symbols per subindustry per sector:
            """)

    """
        #######################

        Securities/stocks per sector and subindustry als list or strings for hovertemplate-annotations:

        #######################
    """

    # https://stackoverflow.com/questions/11869910/pandas-filter-rows-of-dataframe-with-operator-chaining





    print(f"""
    
        #####################################
        Variante 1 Hovertext:
        
        
        Die ndarrays sind schon ganz klasse - die müssten wir nur noch mit np.tolist() in kommagetrennte Listen
        umwandeln:
        #####################################
    
        """)

    sector = 'Energy'
    subindustry = 'Integrated Oil & Gas'

    secs_list = []
    secs_subs_stocks_list = []

    for i in df_all_sectors.sectors.unique():
        print(f"""
                Sector:
                {i}
                """)
        df_single_sec = df_all_sectors[(df_all_sectors.sectors == i)]
        aktien_liste = [i for i in df_single_sec['security']]
        print('Aktien_liste aus Listcomprehension: ', i, 'enthält', len(aktien_liste), 'Aktien. ', aktien_liste)

        secs_list.append(i)
        print('Eine wachsende Liste der Sektoren: ', secs_list)
        for j in df_single_sec.subindustries.unique():
            print(f"""
                            Subindustry:
                            {j}
                            """)
            df_single_sec_single_sub = df_single_sec[df_single_sec.subindustries == j].security.values
            print(df_single_sec_single_sub, type(df_single_sec_single_sub))
            df_single_sec_single_sub_list = df_single_sec_single_sub.tolist()
            print(df_single_sec_single_sub_list, type(df_single_sec_single_sub_list))
            secs_subs_stocks_list.append(df_single_sec_single_sub_list)

    print(f"""

        Wir haben jetzt eine Sektorenliste zum Iterierieren und Symbollisten pro Subind. zum Durchiterieren.

        Hier weiterarbeiten...


        """)

    print('Fertige customdata-Liste: ', secs_subs_stocks_list)

    """
        variante 2 - nur für einen Sektor:
        
    """

    sector = 'Energy'
    subindustry = 'Integrated Oil & Gas'
    subs_stocks_list_energy = []
    # df_single_sector_energy = df_all_sectors[df_all_sectors['sectors'] == 'Energy']

    df_single_sec_energy = df_all_sectors[(df_all_sectors.sectors == 'Energy')]

    # Sort in the right order:
    df_single_sec_energy = df_single_sec_energy[['subindustries', 'security']].sort_values(by='subindustries')
    aktien_liste_energy = [a for a in df_single_sec_energy['security']]
    print('Aktien_liste aus Listcomprehension: aktien_liste_energy enthält', len(aktien_liste_energy), 'Aktien. ',
          aktien_liste_energy)

    for j in df_single_sec_energy.subindustries.unique():
        print(f"""
                                Subindustry:
                                {j}
                                """)
        df_single_sub_energy = df_single_sec_energy[df_single_sec_energy.subindustries == j].security.values
        print(df_single_sub_energy, type(df_single_sub_energy))
        df_single_sub_energy_list = df_single_sub_energy.tolist()
        print(df_single_sub_energy_list, type(df_single_sub_energy_list))
        subs_stocks_list_energy.append(df_single_sub_energy_list)

    print('Fertige customdata-Liste nur Energy - subs_stocks_list_energy: ', subs_stocks_list_energy)

    # Sort in the right order:
    df_single_sector_energy = df_single_sector_energy[['subindustries', 'security']].sort_values(by='subindustries')

    figure3 = go.Figure(data=[
        go.Bar(x=df_single_sector_subindustries_grouped.subindustries, y=df_single_sector_subindustries_grouped.Symbol,
               hovertext=subs_stocks_list_energy,
               marker_color=colors_subs,
               showlegend=False,
               name='Subs'),
        #go.Bar(x=df_sectors_subindustries_grouped['subindustries'].values, y=df_sectors_subindustries_grouped.Symbol,
               #marker_color=colors_subs, name='Subindustries'),
        #go.Bar(x=df_single_sector, y=df_single_sector,
               #marker_color=colors_subs, name='Subindustries'),
    ])

    figure3.update_traces(
        # hovertemplate nimmt leider den gesamten hstring, ohne ihn auf die cols zu verteilen:
        # hovertemplate=(f"<b>{customdata}</b><br>" + "Subindustry: %{x}<br>" + "Number of constituents: %{y}<br>" + "<extra></extra>"),
    )

    figure3.update_layout(barmode='group',
                          hoverlabel=dict(
                                        bgcolor="white",
                                        font_size=16,
                                        font_family="Rockwell"
                                            ),
                          hoverlabel_align='auto'
                          )
    figure3.update_xaxes(title='Fig. 6: Single sector with related Subindustries')
    figure3.update_yaxes(title='Count')
    figure_sec.show()
    figure2.show()
    figure3.show()

    return figure_sec, figure2, figure3


if __name__ == '__main__':
    lineplot_daily_returns_pctchange()
    scatter_plot_ret_vol_sr()
    bar_plot_sr_all_brackets()
    barplot_all_sectors_subindustries()
    #df_returns_per_rating_bracket = main_calculations(['BBB-'], n_portfolios=10000) #['BBB-'],n_portfolios=1000
    print(df_all_sectors)
    print(df_sectors_subindustries_grouped)

    #None












