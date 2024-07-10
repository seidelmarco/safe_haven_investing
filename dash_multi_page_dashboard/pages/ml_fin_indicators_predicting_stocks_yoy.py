import os
import pandas as pd
import numpy as np

# Finance related operations
# Edit 2024: pandas_datareader scheint mit yahoo nicht mehr zu funktionieren - nutze yfinance:
from pandas_datareader import data

import matplotlib.pyplot as plt

import dash

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table, State
import dash_ag_grid as dag

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Todo: comment it out in case you want to start script solely as command-line-app:
# dash.register_page(__name__, path='/model1', title='10 - Financial Data Model 1',
#                    name='10 - Model 1')


def data_collector():
    """

    :return:
    """
    raw_data = pd.read_csv('C:/Users/M.Seidel/PycharmProjects/safe_haven_investing/deep_learning_neural_networks/data_csv_excel/fin_indicators_us_stocks/2018_Financial_Data.csv',
                           index_col=0)      # 'Unnamed: 0'
    #print(raw_data.head())
    #print(raw_data.describe(include='all'))
    #print(raw_data.describe(include='all').T)

    data = raw_data.copy(deep=True)

    # data.rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True) - kann irgendwann gelöscht werden, weil Spalte 0 heißt
    # print(data.index)
    # print(data.columns)
    print(data.axes)


    # GENIAL!
    # print(data.dtypes)
    # print(data.head())
    # print(data.tail())

    # Drop rows with no information:
    data.dropna(how='all', inplace=True)

    return data


def preprocessing():
    """
    Cleaning and cleansing: hier mit outliers, nan, categoricals etc. arbeiten.
    :return:
    """
    df = data_collector()

    print('Rohdaten (var df): \n', df)

    """
    STEP 1: general info, sectors and classes for checking if inbalances lead to later overfitting:
    """

    print(df.info())
    # dtypes: float64(222), int64(1), object(1): 222 indicators, int is the class, obj is categorical (Sector)

    # Look at the class distribution (buy, don't buy) and sector distribution:
    df_class = df['Class'].value_counts()

    fig_class_count = go.Figure(
        data=go.Bar(x=np.arange(len(df_class)), y=df_class, marker_color=(colors['anthrazitgrau'], colors['darkgoldenrod']))  ## marker color can be a single color value or an iterable
    )
    # auch hier lässt sich mit den tick-properties arbeiten:
    fig_class_count.update_layout(
        title={
            "text": f"Class Count ",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                # "family": "Roboto",
                "size": 20,
                "color": colors['slate-grey-darker']
            }
        }
    )
    # https://plotly.com/python/reference/layout/xaxis/
    fig_class_count.update_xaxes(title_text="Don't Buy | Buy",
                                 ticks='outside',
                                 ticklabelmode='period',
                                 tickcolor='black',
                                 ticklen=5,
                                 tickmode='array',
                                 tickvals=[0, 1],
                                 ticktext=('Don\'t buy', 'Buy'),
                                 # Type: dict containing one or more of the keys listed below.
                                 minor=dict(
                                     griddash='dot',
                                     gridcolor='white'
                                 ))
    fig_class_count.update_yaxes(title_text='No. of stocks')
    fig_class_count.show()

    df_sector = df['Sector'].value_counts()
    print(df_sector)
    sec_list = df_sector.index.values.tolist()
    print(sec_list)

    fig_sec_count = go.Figure(
        data=go.Bar(x=np.arange(len(df_sector)), y=df_sector, marker_color=[i for i in px.colors.sequential.ice_r])  # qualitative.Prism, sequential.Sunsetdark   # marker_color=(colors['darkgoldenrod'],colors['dark-blue-grey'])
    )
    fig_sec_count.update_layout(
        title={
            "text": f"Sector Count ",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                # "family": "Roboto",
                "size": 20,
                "color": colors['slate-grey-darker']
            }
        }
    )
    fig_sec_count.update_xaxes(
        ticks='outside',
        tickvals=[i for i in range(len(sec_list))],
        ticktext=[i for i in sec_list],
        tickangle=-45
    )
    fig_sec_count.update_yaxes()
    fig_sec_count.show()

    """
    STEP 2: price variation, look for outliers/errors
    """

    # df = df.reset_index().rename(columns={'index': 'Ticker'})



    # There are comp. like YRIV that have bad data like no revenue (0.00) and so on - how to treat? Delete or
    # taking as training data?
    # for the first try we let them in coz they teach the model to predict not to buy...
    yriv = df[df.index == 'YRIV']
    print('YRIV: \n', yriv)

    # Extract the columns we need in this step from the dataframe
    df_ = df.loc[:, ['Sector', '2019 PRICE VAR [%]']]

    # Get list of sectors
    sector_list = df_['Sector'].unique()

    # Plot the percent price variation for each sector
    for sector in sector_list:
        temp = df_[df_['Sector'] == sector]

        fig_sec_pricevar = go.Figure(
            data=go.Scatter(y=temp['2019 PRICE VAR [%]'])
        )

        fig_sec_pricevar.update_layout(
            title={
                "text": f"{sector.upper()} ",
                "y": 0.95,  # Sets the y position with respect to `yref`
                "x": 0.5,  # Sets the x position of title with respect to `xref`
                "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
                "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
                "font": {  # Only configures font for title
                    # "family": "Roboto",
                    "size": 20,
                    "color": colors['slate-grey-darker']
                }
            }
        )
        fig_sec_pricevar.update_xaxes()

        """
        uncomment when you want to see pricepeaks of all sectors:
        """
        # fig_sec_pricevar.show()

    healthcare_constituents = df[df['Sector'] == 'Healthcare'].index.values.tolist()
    print(healthcare_constituents)

    fig_pricepeaks_healthcare = go.Figure(
        data=go.Scatter(y=df_[df_['Sector'] == 'Healthcare']['2019 PRICE VAR [%]'],
                        ),
    )
    fig_pricepeaks_healthcare.update_layout(
        title={
            "text": f"""HEALTHCARE <br> We see unusual pricepeaks and gonna drop <br> all stocks with changes above 500%""",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                # "family": "Roboto",
                "size": 16,
                "color": colors['slate-grey-darker']
            }
        },
        showlegend=False,
    )
    fig_pricepeaks_healthcare.update_xaxes(
        ticks='outside',
        tickvals=[i for i in range(len(healthcare_constituents))],
        ticktext=[i for i in healthcare_constituents],
        tickangle=-90
    )
    fig_pricepeaks_healthcare.show()

    """
    dropping the pricepeaks:
    
    Are the pricepeaks organic? We'll be checking the price-timeseries of one year to see the days
    that cause the peaks with our own eyes:
    """

    # Get stocks that increased more than 500%
    gain = 500
    top_gainers = df_[df_['2019 PRICE VAR [%]'] >= gain]
    top_gainers = top_gainers['2019 PRICE VAR [%]'].sort_values(ascending=False)

    print(f"""{len(top_gainers)} stocks with more than {gain}% gain.""")

    # Top gainers adjclose and volume in 2019:
    date_start = dt.datetime(2019, 1, 1)
    date_end = dt.datetime(2019, 12, 31)

    tickers = top_gainers.index.values.tolist()

    daily_price_all = yf.download(tickers, date_start, date_end)    #pivot table: how to depivot?
    print(daily_price_all)

    for ticker in tickers:
        # Pull daily prices for each ticker from Yahoo Finance
        daily_price = yf.download(tickers=ticker, start=date_start, end=date_end)

        # Plot prices with volume
        # add trace
        fig_pricedata = make_subplots(2, 1, print_grid=True, subplot_titles=['Adj Close', 'Volume'], y_title='Volume and Adj Close in $',
                                      vertical_spacing=.1, shared_xaxes=False)       # specs=[[{'rowspan':2}], [{}], [{'rowspan':1}]],
        fig_pricedata.add_trace(go.Scatter(x=daily_price.index, y=daily_price['Adj Close'], name='Adj CLose', marker_color='dodgerblue'), 1, 1)
        fig_pricedata.add_trace(go.Scatter(x=daily_price.index, y=daily_price['Volume'], name='Volume', marker_color='darkgoldenrod'), 2, 1)

        fig_pricedata.update_layout(title={'text': f"""{ticker} 1 of {len(top_gainers)} stocks with more than {gain}% gain.""",
                                           "y": 0.95,  # Sets the y position with respect to `yref`
                                           "x": 0.5,  # Sets the x position of title with respect to `xref`
                                           "xanchor": "center",
                                           # Sets the title's horizontal alignment with respect to its x position
                                           "yanchor": "top",
                                           # Sets the title's vertical alignment with respect to its y position. "
                                           })
        fig_pricedata.update_xaxes()
        # fig_pricedata.show()

    # Delete stocks that don't make sense from df
    inorganic_stocks = ['HEBT', 'ARQL', 'DRIO', 'SSI', 'ANFI', 'DRRX', 'AXSM', 'ALIM'] #axsm, alim lassen wir drin - axsm hat 3500%; Edit: doch raus, weil es ja um die 19er Daten und das Modell geht, nicht wie sich die Daten 2024 entwickelt haben

    """
    STEP 2.1 repetition: price variation, w/o inorganic stocks
    """
    df.drop(inorganic_stocks, axis=0, inplace=True)

    # Extract the columns we need in this step from the dataframe
    df_organics = df.loc[:, ['Sector', '2019 PRICE VAR [%]']]

    # Plot the percent price variation for each sector
    for sector in sector_list:
        temp = df_organics[df_organics['Sector'] == sector]

        fig_sec_pricevar = go.Figure(
            data=go.Scatter(y=temp['2019 PRICE VAR [%]'])
        )

        fig_sec_pricevar.update_layout(
            title={
                "text": f"{sector.upper()} ",
                "y": 0.95,  # Sets the y position with respect to `yref`
                "x": 0.5,  # Sets the x position of title with respect to `xref`
                "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
                "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
                "font": {  # Only configures font for title
                    # "family": "Roboto",
                    "size": 20,
                    "color": colors['slate-grey-darker']
                }
            }
        )
        fig_sec_pricevar.update_xaxes()

        """
        uncomment when you want to see pricepeaks of all sectors:
        """
        # fig_sec_pricevar.show()

    ##################################################
    healthcare_constituents_organics = df_organics[df_organics['Sector'] == 'Healthcare'].index.values.tolist()

    fig_nopricepeaks_healthcare = go.Figure(
        data=go.Scatter(y=df_organics[df_organics['Sector'] == 'Healthcare']['2019 PRICE VAR [%]'],
                        ),
    )
    fig_nopricepeaks_healthcare.update_layout(
        title={
            "text": f"""HEALTHCARE <br> Cleansed without unorganic pricepeaks""",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                # "family": "Roboto",
                "size": 16,
                "color": colors['slate-grey-darker']
            }
        },
        showlegend=False,
    )
    fig_nopricepeaks_healthcare.update_xaxes(
        ticks='outside',
        tickvals=[i for i in range(len(healthcare_constituents_organics))],
        ticktext=[i for i in healthcare_constituents_organics],
        tickangle=-90
    )
    fig_nopricepeaks_healthcare.show()

    highest_gainer = df_organics[df_organics['2019 PRICE VAR [%]'] > 500]   # 3000 nur axsm
    print('Highest: \n', highest_gainer)

    """
    STEP 3: HANDLE MISSING VALUES, 0-VALUES
    
    The next check we need to perform concerns the presence of missing values (NaN). At the same time, 
    I think it is also useful to check the quantity of 0-valued entries. 
    What I like to do is simply plot a bar chart of the count of both missing values and 0-valued entries, in order to take a first look at the situation. 
    (Due to the large quantity of financial indicators available, I will make quite a big plot)

    Before doing that, we can drop the categorical columns from the dataframe df, since we won't be needing them now.
    """

    # Drop columns relative to classification, we will use them later
    class_data = df.loc[:, ['Sector', '2019 PRICE VAR [%]']]
    df_wo_class = df.drop(['Sector', '2019 PRICE VAR [%]'], axis=1)

    # Plot initial status of data quality in terms of nan-values and zero-values
    nan_vals = df_wo_class.isna().sum()
    zero_vals = df_wo_class.isin([0]).sum()
    ind = np.arange(df_wo_class.shape[1])

    fig_nans_zeros = make_subplots(2, 1, print_grid=True, subplot_titles=['Nan-Values Count', 'Zero-Values Count'],
                                  y_title='Nans and Zeros',
                                  vertical_spacing=.5,
                                  shared_xaxes=False)  # specs=[[{'rowspan':2}], [{}], [{'rowspan':1}]],
    fig_nans_zeros.add_trace(
        go.Bar(x=ind, y=nan_vals.values.tolist(), name='Nans', marker_color='dodgerblue'), 1, 1)
    fig_nans_zeros.add_trace(
        go.Bar(x=ind, y=zero_vals.values.tolist(), name='Zeros', marker_color='darkgoldenrod'), 2, 1)

    fig_nans_zeros.update_layout(
        title={
            "text": f"INITIAL INFORMATION ABOUT DATASET ",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                # "family": "Roboto",
                "size": 20,
                "color": colors['slate-grey-darker']
            }
        }
    )
    fig_nans_zeros.update_xaxes(
        ticks='outside',
        tickvals=[i for i in range(len(nan_vals.index.tolist()))],
        ticktext=[i for i in nan_vals.index.tolist()],
        tickangle=-90
    )
    fig_nans_zeros.update_yaxes()

    fig_nans_zeros.show()

    """
    ESTABLISH THE DOMINANCE LEVEL of NANS and ZEROS

    We can see that:

    There are quite a lot of missing values
    There are also a lot of 0-valued entries. For some financial indicators, almost every entry is set to 0.

    To understand the situation from a more quantitative perspective, it is useful to count the occurrences of 
    both missing-values and 0-valued entries, and sort them in descending order. This allows us to establish 
    the dominance level for both missing values and 0-valued entries
    """

    # Find count and percent of nan-values, zero-values
    total_nans = nan_vals.sort_values(ascending=False)

    percent_nans = (df_wo_class.isna().sum() / df_wo_class.isna().count() * 100).sort_values(ascending=False)
    total_zeros = zero_vals.sort_values(ascending=False)
    percent_zeros = (df_wo_class.isin([0]).sum() / df_wo_class.isin([0]).count() * 100).sort_values(ascending=False)
    df_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])
    df_zeros = pd.concat([total_zeros, percent_zeros], axis=1, keys=['Total Zeros', 'Percent Zeros'])
    print('df_nans \n', df_nans)
    print('df_zeros \n', df_zeros)

    # Graphical representation

    fig_nan_dominance = go.Figure(
        data=go.Bar(x=np.arange(30), y=df_nans['Percent NaN'].iloc[:30].values.tolist(), marker_color=px.colors.cyclical.IceFire)   # namespaces: cyclical, diverging, qualitative, sequential
    )
    fig_nan_dominance.update_layout()
    fig_nan_dominance.update_xaxes(
        ticks='outside',
        tickvals=[i for i in range(len(df_nans.iloc[:30].index.tolist()))],
        ticktext=[i for i in df_nans.iloc[:30].index.tolist()],
        tickangle=-90,
        minor=dict(
            griddash='dot',
            gridcolor='white'
        )
    )
    fig_nan_dominance.update_yaxes(title_text='NAN-Dominance [%]')
    fig_nan_dominance.show()

    fig_zero_dominance = go.Figure(

        # oder x= df_zeros.index
        data=go.Bar(x=np.arange(30), y=df_zeros['Percent Zeros'].iloc[:30].values.tolist(), marker_color=px.colors.sequential.ice)
    )
    fig_zero_dominance.update_layout()
    fig_zero_dominance.update_xaxes(
        ticks='outside',
        tickvals=[i for i in range(len(df_zeros.iloc[:30].index.tolist()))],
        ticktext=[i for i in df_zeros.iloc[:30].index.tolist()],
        tickangle=-90,
        minor=dict(
            griddash='dot',
            gridcolor='white'
        )
    )
    fig_zero_dominance.update_yaxes(title_text='ZEROS-Dominance [%]')
    fig_zero_dominance.show()

    """
    The two plots above clearly show that to improve the quality of the dataframe df we need to:

    fill the missing data
    fill or drop those indicators that are heavy zeros-dominant.

    What levels of nan-dominance and zeros-dominance are we going to tolerate?
    
    I usually determine a threshold level for both nan-dominance and zeros-dominance, which corresponds to a given 
    percentage of the total available samples (rows): if a column has a percentage of nan-values and/or zero-valued 
    entries higher than the threshold, I drop it.
    
    For this specific case we know that we have 4392 samples, so I reckon we can set:

    nan-dominance threshold = 5-7% - 1 Model: we start with 6% - so we delete cols/indicators with more than 263 Nans
    zeros-dominance threshold = 5-10% - 1 Model: we start with 7,5% - so we delete cols/indicators with more than 329 Zeros
    
    Once the threshold levels have been set, I iteratively compute the .quantile() of both df_nans and df_zeros in order 
    to find the number of financial indicators that I will be dropping. In this case, we can see that:

    We need to drop the top 50% (test_nan_level=1-0.5=0.5) nan-dominant financial indicators in order to not have 
    columns with more than 226 nan values, which corresponds to a nan-dominance threshold of 5.9% (aligned with our 
    initial guess).
    We need to drop the top 40% (test_zeros_level=1-0.4=0.6) zero-dominant financial indicators in order to not have 
    columns with more than 283 0 values, which corresponds to a zero-dominance threshold of 7.5% 
    (aligned with our initial guess).


    """

    # Find reasonable threshold for nan-values situation
    test_nan_level = 0.5
    print(df_nans.quantile(test_nan_level))
    # _ is just the first var (we won't need it)
    _, thresh_nan = df_nans.quantile(test_nan_level)
    print('Thresh nan: \n', _, thresh_nan)

    # Find reasonable threshold for zero-values situation
    test_zeros_level = 0.6
    print(df_zeros.quantile(test_zeros_level))
    _, thresh_zeros = df_zeros.quantile(test_zeros_level)
    print('Thresh zeros: \n', thresh_zeros)

    """
    Once the threshold levels have been set, I can proceed and drop from df those columns (financial indicators) that 
    show dominance levels higher than the threshold levels, in terms of both missing values and 0-valued entries.
    
    So, we reduced the number of financial indicators available in the dataframe df to 62. By doing so, we removed all 
    those columns characterized by heavy nan-dominance and zeros-dominance.
    
    We should always keep in mind that this is quite a brute force approach, and there is the possibilty of 
    having dropped useful information.
    """

    # Clean dataset applying thresholds for both zero values, nan-values
    print(f'INITIAL NUMBER OF VARIABLES: {df_wo_class.shape[1]}')
    print()

    df_test1 = df_wo_class.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, axis=1)
    print(f'NUMBER OF VARIABLES AFTER NaN THRESHOLD {thresh_nan:.2f}%: {df_test1.shape[1]}')
    print()

    df_zeros_postnan = df_zeros.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, axis=0)
    df_test2 = df_test1.drop((df_zeros_postnan[df_zeros_postnan['Percent Zeros'] > thresh_zeros]).index, axis=1)
    print(f'NUMBER OF VARIABLES AFTER Zeros THRESHOLD {thresh_zeros:.2f}%: {df_test2.shape[1]}')
    print()
    print("""
    
        df_test2:
    
    """)
    print(df_test2)
    print()

    """
    STEP 4: CORRELATION MATRIX, CHECK MISSING VALUES AGAIN
    
    The correlation matrix is an important tool that can be used to quickly evaluate the linear correlation between 
    variables, in this case financial indicators. As clearly explained here, a positive linear correlation value 
    between two variables means that they move in a similar way; a negative linear correlation value between two 
    variables means that they move in opposite ways. Finally, if the correlation value is close to 0, 
    then their trends are not related.

    Looking at the figure below, we can see that there is a chunk of financial indicators that show no linear 
    correlation whatsoever. Those financial indicators are the heavy nan-dominant ones (as highlighted in 
    the barplot below). This means that this chart will change once we will fill the nan values.
    """

    # Plot correlation matrix
    fig_corr_matrix = px.imshow(df_test2.corr(), color_continuous_scale='YlGnBu',
                                # title='Correlation Matrix of Output Dataset',
                                zmin=-1, zmax=1,
                                # aspect="auto"
                                )       # 'RdBu', YlGnBu px.colors.diverging.Picnic
    fig_corr_matrix.show()

    """
    We can evaluate the impact of our choices in terms of threshold levels by plotting again the count of missing 
    values and 0-valued entries occurring in the remaining financial indicators. The situation has clearly improved, 
    even if a few financial indicators maintain high levels of nan-dominance, which is evident when looking at 
    the correlation matrix above.
    """

    # New check on nan values
    fig_nan_dominance_new = make_subplots(2, 1, shared_xaxes=True, print_grid=True, vertical_spacing=0.2)
    # df_test2.columns - brings sum and count the same outcome?
    fig_nan_dominance_new.add_trace(go.Bar(x=np.arange(df_test2.shape[1]), y=df_test2.isna().sum(),
                                           marker_color='dodgerblue', name='Nans'), 1, 1)
    fig_nan_dominance_new.add_trace(go.Bar(x=np.arange(df_test2.shape[1]), y=df_test2.isin([0]).sum(),
                                           marker_color='darkgoldenrod', name='Zeros'), 2, 1)
    # wenn ich 1, 1 und 2, 1 nicht setze, sehe ich die Balken gruppiert nebeneinander :-) ABER ich sehe keinen ticktext
    fig_nan_dominance_new.update_layout(title='INFORMATION ABOUT DATASET - CLEANED NAN + ZEROS')
    fig_nan_dominance_new.update_xaxes(
        ticks='outside',
        tickvals=[i for i in range(len(df_test2.columns))],
        ticktext=[i for i in df_test2.columns.tolist()],
        tickangle=-90,
        minor=dict(
            griddash='dot',
            gridcolor='white'
        )
    )
    fig_nan_dominance_new.show()

    """
    STEP 5: HANDLE EXTREME VALUES
    Analyzing the df with the method .describe() we can see that some financial indicators show a large discrepancy 
    between max value and 75% quantile. Furthermore, we also have standard deviation values that are very large! 
    This could be a sign of the presence of outliers: to be conservative, I will drop the top 3% and bottom 3% of 
    the data for each financial indicator.
    """

    # Analyze dataframe
    print(df_test2.describe())

    """
    We see:
    - Revenue growth between 75% and max completely exxagerated
    - huge 75% and max discrepancies
    - huge stdeviations!
    - we have to cut outliers
    - to be conservative, I will drop the top 3% and bottom 3% of the data for each financial indicator.
    """

    # Cut outliers:
    # it yields two df with tickers as index, indicators are columns and values are True/False
    top_quantiles = df_test2.quantile(0.97)
    outliers_top = (df_test2 > top_quantiles)
    print('DF outliers top: \n', outliers_top)

    low_quantiles = df_test2.quantile(0.03)
    outliers_low = (df_test2 < low_quantiles)
    print('DF outliers low: \n', outliers_low)

    df_w_nans = df.reset_index().rename(columns={'index': 'Ticker'})

    # mask function
    # The mask() method replaces the values of the rows (Set to NaN) where the condition evaluates to True.
    #
    # The mask() method is the opposite of the where() method.
    # returns a DataFrame with the result, or None if the inplace parameter is set to True.
    # where the condition is false, we keep the original value
    """
    Example:
    all even will be replaced by 0
    import pandas as pd

    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
    })

    # use mask() to replace even values 
    # across the entire DataFrame with 0
    df = df.mask(df % 2 == 0, 0)
    
    print(df)
    """
    df_test2 = df_test2.mask(cond=outliers_top, other=top_quantiles, axis=1)
    df_test2 = df_test2.mask(cond=outliers_low, other=low_quantiles, axis=1)

    # Take a look at the dataframe post-outliers cut
    print(df_test2.describe())

    """
    Looking at the statistical description of the dataframe df post outliers removal, we can see that we managed to 
    decrease the standard deviation values considerably, and also the discrepancy between max value and 75% quantile 
    is smaller.
    
    
    STEP 6: FILL MISSING VALUES

    We can now fill the missing values, but how? There are several methods we could use to fill the missing values:

    fill nan with 0
    fill nan with mean value of column
    fill nan with mode value of column
    fill nan with previous value (bfill or ffill)
    ...

    In this case, I think it is appropriate to fill the missing values with the mean value of the column. However, 
    we must not forget the intrinsic characteristics of the data we are working with: we have a many stocks from many 
    different sectors. It is fair to expect that each sector is characterized by macro-trends and macro-factors 
    that may influence some financial indicators in different ways. 
    So, I reckon that we should keep this separation somehow.

    From a practical perspective, this translates into filling the missing value with the mean value 
    of the column, grouped by each sector.
    """

    # Add the sector column
    df_out = df_test2.join(df['Sector'])
    print(df_out)

    # Replace nan-values with mean value of column, considering each sector individually.
    # apply() vs transform(): Apply() returns a new DataFrame while Transform() returns a series
    df_out = df_out.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))

    # Once that's done, we can plot again the correlation matrix in order to evaluate the impact of our choices.

    # Plot correlation matrix
    fig_corr_matrix2 = px.imshow(df_out.corr(), color_continuous_scale='YlGnBu',
                                # title='Correlation Matrix of Output Dataset',
                                zmin=-1, zmax=1,
                                # aspect="auto"
                                )  # 'RdBu', YlGnBu px.colors.diverging.Picnic
    fig_corr_matrix2.show()

    """
    As we can see, the chunk of financial indicators that was characterized by linear correlation equal to 0 is 
    now more organic (i.e. correlation is either positive or negative), thanks to the fact that we replaced 
    the missing values with the respective mean value of the column (per-sector).
    
    STEP 7: ADD TARGET DATA
    """
    # Add back the classification columns
    df_out = df_out.join(class_data)

    print('Final Dataframe df_test2: \n', df_out)

    # Print information about dataset
    df_out.info()
    print(df_out.describe())












    """
    STEP 3: Treat the nans:
    
    dropna, fillna, bfill....
    
    available_stocks_only_ratings.dropna(axis=0, subset=['LT-Rating_orig'], inplace=True)
    """




    """
    Slice the 1% - 99% percentile:
    """




    rowdata = df_w_nans.to_dict("records")
    columns = [{'field': j, 'name': j} for j in df_w_nans.columns]

    # Initial choice of indicators:
    # print(df_w_nans)
    df_features = df_w_nans[['Revenue', 'Revenue Growth', 'EPS', 'priceEarningsRatio', 'Dividend Yield',
                                'Free Cash Flow margin', 'Earnings Before Tax Margin',
                                'EBIT Growth', 'EPS Growth', 'Dividends per Share Growth',
                                'Sector', '2019 PRICE VAR [%]',
                                'Class']]

    #print(df_features)

    columns_choice = [{'field': 'Ticker', 'name': 'Symbol'}] + [{'field': j, 'name': j} for j in df_features.columns]

    return_dictionary = {
        'class_data': class_data,
        'df_wo_class': df,
        'df_cleaned': df_w_nans,
        'rowdata': rowdata,
        'columns': columns,
        'df_features': df_features,
        'columns_choice': columns_choice,
        'fig_class_count': fig_class_count,
        'fig_sec_count': fig_sec_count,
        'fig_pricepeaks_healthcare': fig_pricepeaks_healthcare,
        'fig_nopricepeaks_healthcare': fig_nopricepeaks_healthcare,
    }
    return return_dictionary


return_dict = preprocessing()


def indicators_model():
    """
    Meine Modellidee - alle Features auf 5 Punkte prüfen...

    Check for OLS-Assumptions...

    y_ttm_hat = earnings + revenue + costs + divausschüttungsquote + debts + free cash flow,dcf, um den risiko-
     freien Zins mit reinzubringen etc.., beta, p-e-ratio, yield
    :return:
    """


"""
----------------------------------
Plotting:
----------------------------------
"""


def histograms_raw(feature: str = 'Dividend Yield'):
    """

    :return:
    """
    figure_hist_raw = go.Figure(
        data=go.Histogram(x=return_dict['df_features'][feature])
    )
    figure_hist_raw.update_layout(
        title={
            "text": f"Comparison of feature-distribution of {feature} before preprocessing/cutting outliers: ",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                "family": "Roboto",
                "size": 16,
                "color": colors['slate-grey-darker']
            }
        }
    )
    figure_hist_raw.update_xaxes(title_text=feature)
    figure_hist_raw.update_yaxes(title_text='Count')
    figure_hist_raw.show()

    fig_overlaid = go.Figure()
    fig_overlaid.add_trace(go.Histogram(x=return_dict['df_features']['Revenue']))
    fig_overlaid.add_trace(go.Histogram(x=return_dict['df_features']['Revenue Growth']))

    # Overlay both histograms
    fig_overlaid.update_layout(barmode='overlay')       # 'stack'
    # Reduce opacity to see both histograms
    fig_overlaid.update_traces(opacity=0.75)
    # fig_overlaid.show()

    fig_individual_bins = go.Figure()

    fig_individual_bins.add_trace(go.Histogram(
        x=return_dict['df_features']['Revenue'],
        name='Revenue',
        xbins=dict(
            start=1000000,
            end=100000000,
            size=1000000
        ),
        marker_color='#EB89B5',
        opacity=0.75
    ))

    fig_individual_bins.update_layout(
        title_text='Sampled Results',  # title of plot
        xaxis_title_text='Revenue in Dollar',  # xaxis label
        yaxis_title_text='Count',  # yaxis label
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )
    fig_individual_bins.show()

    return figure_hist_raw


def histograms_not_preprocessed():
    """

    :return:
    """
    figure_subplots = make_subplots(rows=4, cols=2, shared_yaxes=False,
                                    start_cell="top-left",
                                    print_grid=True,
                                    horizontal_spacing=None,
                                    vertical_spacing=None,
                                    subplot_titles=None,
                                    column_widths=None,
                                    row_heights=None,
                                    specs=None,
                                    insets=None,
                                    column_titles=None,
                                    row_titles=None,
                                    x_title=None,
                                    y_title='Count',
                                    )

    trace1 = go.Histogram(x=return_dict['df_features']['Revenue'], name='Revenue', nbinsx=20)
    trace2 = go.Histogram(x=return_dict['df_features']['Revenue Growth'], name='Revenue Growth', nbinsx=20)
    trace3 = go.Histogram(x=return_dict['df_features']['EPS'], name='EPS', nbinsx=20)
    trace4 = go.Histogram(x=return_dict['df_features']['priceEarningsRatio'], name='priceEarningsRatio')
    trace5 = go.Histogram(x=return_dict['df_features']['Dividend Yield'], name='Dividend Yield')
    trace6 = go.Histogram(x=return_dict['df_features']['Earnings Before Tax Margin'], name='Earnings Before Tax Margin')
    trace7 = go.Histogram(x=return_dict['df_features']['Free Cash Flow margin'], name='Free Cash Flow margin')
    trace8 = go.Histogram(x=return_dict['df_features']['2019 PRICE VAR [%]'], name='2019 PRICE VAR [%]')

    figure_subplots.add_trace(trace1, 1, 1)
    figure_subplots.add_trace(trace2, 1, 2)
    figure_subplots.add_trace(trace3, 2, 1)
    figure_subplots.add_trace(trace4, 2, 2)
    figure_subplots.add_trace(trace5, 3, 1)
    figure_subplots.add_trace(trace6, 3, 2)
    figure_subplots.add_trace(trace7, 4, 1)
    figure_subplots.add_trace(trace8, 4, 2)

    figure_subplots.update_layout(
        title_text='Distribution before cleaning and cleansing: ',
        autosize=True,
        # autosize determines width and height, but height overrides...
        # so we don't need to use the style-property in layout
        height=800,
    )

    figure_subplots.update_yaxes(automargin=True)

    figure_subplots.show()

    return figure_subplots


def histograms_preprocessed():
    """

    :return:
    """
    figure_subplots = make_subplots(rows=4, cols=2, shared_yaxes=False,
                                    start_cell="top-left",
                                    print_grid=True,
                                    horizontal_spacing=None,
                                    vertical_spacing=None,
                                    subplot_titles=None,
                                    column_widths=None,
                                    row_heights=None,
                                    specs=None,
                                    insets=None,
                                    column_titles=None,
                                    row_titles=None,
                                    x_title=None,
                                    y_title='Count',
                                    )

    trace1 = go.Histogram(x=return_dict['df_features']['Revenue'], name='Revenue', nbinsx=20)
    trace2 = go.Histogram(x=return_dict['df_features']['Revenue Growth'], name='Revenue Growth', nbinsx=20)
    trace3 = go.Histogram(x=return_dict['df_features']['EPS'], name='EPS', nbinsx=20)
    trace4 = go.Histogram(x=return_dict['df_features']['priceEarningsRatio'], name='priceEarningsRatio')
    trace5 = go.Histogram(x=return_dict['df_features']['Dividend Yield'], name='Dividend Yield')
    trace6 = go.Histogram(x=return_dict['df_features']['Earnings Before Tax Margin'], name='Earnings Before Tax Margin')
    trace7 = go.Histogram(x=return_dict['df_features']['Free Cash Flow margin'], name='Free Cash Flow margin')
    trace8 = go.Histogram(x=return_dict['df_features']['2019 PRICE VAR [%]'], name='2019 PRICE VAR [%]')

    figure_subplots.add_trace(trace1, 1, 1)
    figure_subplots.add_trace(trace2, 1, 2)
    figure_subplots.add_trace(trace3, 2, 1)
    figure_subplots.add_trace(trace4, 2, 2)
    figure_subplots.add_trace(trace5, 3, 1)
    figure_subplots.add_trace(trace6, 3, 2)
    figure_subplots.add_trace(trace7, 4, 1)
    figure_subplots.add_trace(trace8, 4, 2)

    figure_subplots.update_layout(
        title_text='Distribution after cleaning and cleansing: ',
        autosize=True,
        # autosize determines width and height, but height overrides...
        # so we don't need to use the style-property in layout
        height=800,
    )

    figure_subplots.update_yaxes(automargin=True)

    figure_subplots.show()

    return figure_subplots


def scatter_plot_revenue():
    """

    :return:
    """

    figure_scat = go.Figure(
        data=go.Scatter(x=return_dict['df_features']['Revenue'], y=return_dict['df_features']['2019 PRICE VAR [%]'], mode='markers',
                                       marker=dict(
                                           size=10,
                                           color=return_dict['df_features']['Class'],
                                           colorscale='Viridis',
                                           colorbar=dict(title='Class'),
                                           showscale=True,
                                           line_width=1,
                                           line_color='black'
                                       ),
                                       #name='Sharpe Ratio',
                                       showlegend=False,
                                       hovertext=[i for i in return_dict['df_features']['Sector']],
                                       hoverinfo='text',)
    )
    # figure_scat.add_trace()
    figure_scat.update_layout(
        font={
            'family': 'Roboto',
            'size': 16
        },
        # Update title font
        title={
            "text": f"Distribution of pricechanges:",
            "y": 0.95,  # Sets the y position with respect to `yref`
            "x": 0.5,  # Sets the x position of title with respect to `xref`
            "xanchor": "center",  # Sets the title's horizontal alignment with respect to its x position
            "yanchor": "top",  # Sets the title's vertical alignment with respect to its y position. "
            "font": {  # Only configures font for title
                "family": "Roboto",
                "size": 16,
                "color": colors['slate-grey-darker']
            }
        }
    )
    figure_scat.update_xaxes(title_text='Revenue')
    figure_scat.update_yaxes(title_text='Pricechange in %')

    figure_scat.show()
    return figure_scat


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
                                rowData=return_dict['rowdata'],
                                columnDefs=return_dict['columns'],
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
                    dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.Br(),
                            html.Div(children='Indicators for the first model: ',
                                     style={'textAlign': 'center'}),
                            html.Br(),
                            html.Hr(),
                            dag.AgGrid(
                                id='table-financial-data-indicators',
                                rowData=return_dict['rowdata'],
                                columnDefs=return_dict['columns_choice'],
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

                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            html.Div([
                                dcc.Markdown([
                                    '1. Categorical variables',
                                    '- The samples are not balanced in terms of class. \n',
                                    '- This should be accounted for when splitting the data between training and testing data. \n',
                                    '- Use the stratify-option within sklearn.model_selection.train_test_split\n',
                                    '- There is a total of 11 sectors, 5 of them with about 500+ stocks each, while the remaining 6 sectors have less than 300 stocks. In particular, the sectors Utilities and Communication Services have around 100 samples. This has to be kept in mind if we want to use this data with ML algorithms: there are very few samples, which could lead to overfitting, etc.\n'
                                ]),
                            ]),
                        ])  # end col
                    ]),  # end row

                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='class-count',
                                figure=return_dict['fig_class_count']
                            )
                        ], width=6),
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='sector-count',
                                figure=return_dict['fig_sec_count']
                            )
                        ], width=6),

                    ]),  # end row

                    ################
                    # Making pricepeaks visible and drop those stocks from the dataset:
                    ################
                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='pricepeaks-healthcare',
                                figure=return_dict['fig_pricepeaks_healthcare']
                            )
                        ], width=6),
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='no-pricepeaks-healthcare',
                                figure=return_dict['fig_nopricepeaks_healthcare']
                            )
                        ], width=6),

                    ]),  # end row



                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(id='dist_raw_subplots', figure=histograms_not_preprocessed(),
                                      )
                        ], width=12),
                    ]), # end row

                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(id='dist_clean_subplots', figure=histograms_preprocessed(),
                                      )
                        ], width=12),
                    ]),  # end row

                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            html.Plaintext(
                                'We see the distributions of the cleaned and cleansed Data.\n'
                                'Later we got to take into consideration if sectors with an excess of stocks \n'
                                'lead to overfitting. Also watch the span of the percentiles - are they tight enough?',
                                style={'color': 'green'}
                            ),
                            html.Div([
                                dcc.Markdown([
                                    '2. Price variation, look for outliers/errors',
                                    '- Check if the target data makes sense: \n',
                                    '- Does the PRICE VAR [%] contain any mistake (for instance mistypings or unreasonable values). A quick plot of this column (for each sector), will allow us to assess the situation. \n',
                                    '- In layman\'s terms, here we are looking for major peaks/valleys, which indicate stocks that increased/decreased in value by an incredible amount with respect to the overall sector\'s trend.\n',
                                    '3. ',
                                    'The next check we need to perform concerns the presence of missing values (NaN). At the same time, I think it is also useful to check the quantity of 0-valued entries. What I like to do is simply plot a bar chart of the count of both missing values and 0-valued entries, in order to take a first look at the situation. (Due to the large quantity of financial indicators available, I will make quite a big plot) Before doing that, we can drop the categorical columns from the dataframe df, since we won\'t be needing them now.'
                                ]),
                            ]),
                        ])  # end col
                    ]), # end row

                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='indicator-revenue',
                                figure=scatter_plot_revenue()
                            )
                        ], width=4),
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='indicator-earnings',
                                figure=go.Figure()
                            )
                        ], width=4),
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='indicator-dividend',
                                figure=go.Figure()
                            )
                        ], width=4),
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='indicator-eps',
                                figure=go.Figure()
                            )
                        ], width=4),
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='indicator-fcf-margin',
                                figure=go.Figure()
                            )
                        ], width=4),
                        dbc.Col([
                            html.Br(),
                            dcc.Graph(
                                id='indicator-revenue-growth',
                                figure=go.Figure()
                            )
                        ], width=4),
                    ]), # end row
        ]),  # end dbc.Container

    return layout_model


if __name__ == '__main__':
    """
    Warum werde trotzdem alle Grafiken angezeigt (Funktionen ausgeführt), selbst wenn hier nichts gecalled wird???
    """
    # data_collector()

    # preprocessing()
    # indicators_model()
    # scatter_plot_revenue()
    # histograms_raw()
    # histograms_raw(feature='priceEarningsRatio')
    # histograms_raw(feature='Earnings Before Tax Margin')
    # histograms_raw(feature='Revenue')
    # histograms_not_preprocessed()
    None

