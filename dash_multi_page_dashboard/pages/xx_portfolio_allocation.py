import dash

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table

import plotly.express as px

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

import datetime as dt

# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': None,
    'min_rows': None,
    'max_rows': 20,
    'precision': 8,
    'float_format': lambda x: f'{x:.4f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)


# dash.register_page(__name__, path='/portfolio', title='Portfolio Allocation', name='Portfolio Allocation')


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


def portfolio_allocation():
    """
    https://www.chegg.com/homework-help/questions-and-answers/traceback-recent-call-last-file-pandas-libs-tslibs-parsingpyx-line-440-pandaslibstslibspar-q109763903
    :return:
    """

    # Step 1: Download daily prices for the five stocks
    symbols = ['CMCL', 'TDG', 'PCAR', 'TRGP', 'CARR']
    # quick prototyping:
    #symbols = ['CMCL', 'TRGP']
    #start_date = '2010-01-01'

    # the earliest date when we got data for all five symbols...
    start_date = '2020-03-20'
    end_date = dt.datetime.now()

    prices_df = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    print("""
        Adjclose several tickers.....
        Todo: muss ich die NaNs nicht noch fillen oder droppen??? Ja. wir droppen bei den returns... below-mentioned
    """, prices_df)

    # Step 2: Calculate daily returns of the stocks
    """
    Bsp. CMCL und TRGP 14.11. minus 13.11.23
    CMCL: ln(12,21) - ln(11,50) = 0,0599
    TRGP: ln(86,43) - ln(84,91) = 0,0177
    
    Muss ich ln oder kann ich nicht gleich mit % arbeiten....?
    Nein: wenn ich den aktuellen Tag durch den Vortag teile bekomme ich die % und die sind exakt gleich zur Differenz
    der LN-Gleichungen ;-) 
    
    """
    normalized_returns_df = np.log(prices_df) - np.log(prices_df.shift(1))
    normalized_returns_df.dropna(inplace=True)
    print(f"""
        Normalized daily returns...NaNs got dropped.., only 926 rows/days - starts at 2020-03-20
        {normalized_returns_df}
    """)

    # Step 3: Download risk-free rates
    # Trick: doppelte Klammer, um die pd.series sofort in einen DF mit Spaltenname zu verwandeln

    risk_free_df = yf.download('^IRX', start=start_date, end=end_date)[['Adj Close']]

    print("""
    
    Risk free interest (10 Jahre US-Treasuries) in %:
    
    Trick: die Series in einen DF mit [[]] verwandeln
    """, risk_free_df)
    info = type(risk_free_df)
    print('Risk free DF columns:', info)
    risk_free_df_resampled = risk_free_df.resample('D').last() #.last().ffill().pct_change()
    risk_free_df_resampled_ffill = risk_free_df_resampled.ffill()
    risk_free_df_resampled_ffill_pctchange = risk_free_df_resampled_ffill.pct_change()
    print(f"""
        Resampled risk free rates:
        resampled auf 'D', last()
            https://towardsdatascience.com/resample-function-of-pandas-79b17ec82a78
            the resample('D')-method filled our data with all absolute days - meaning even the weekends:
            resample() goes hand in hand with .last() for that we see the data whilst using print()...
        {risk_free_df_resampled}
        The last() method returns the last n rows, based on the specified value.
        The ffill() method replaces the NULL values with the value from the previous row (or previous column, 
        if the axis parameter is set to 'columns').
        {risk_free_df_resampled_ffill}
        Percent-change... Example 22.11.23: 5.2650% / 5.2550% = 1.0019 = 0.19 % change - our risk free rate was rising
        {risk_free_df_resampled_ffill_pctchange}
        
    """)
    risk_free_df_annualized = risk_free_df_resampled_ffill_pctchange * 252  # Annualize risk-free rates
    risk_free_df_annualized.dropna(inplace=True)
    print("""
        Resampled annualized risk free rates: - we assumed 252 business days (usual 250 - 260)
        ... if you check the numbers with your calculator be aware that python does the math with many more digits
        after the floating point than we see in the terminal
        https://medium.com/cloudcraftz/fundamentals-of-returns-and-its-python-implementations-4db5b8c9bb60
        Annualized risk-free return means, what if you had gotten risk free pctchange every day for all 252
        business days - so 0.0019 leads to 0.4796 (47.96 %)
    """)
    print(risk_free_df_annualized)
    print(input('Stop here...'))

    # Step 4: Compute excess returns
    # here we want to substract the risk free interest from the daily returns but we cant parse IRX...
    excess_returns_df = normalized_returns_df.sub(risk_free_df_annualized['Adj Close'], axis=0)       #['^IRX'], axis=0)

    print("""
            Excess Returns DF (normalized_df substracted by risk free annualized):
            
            Ich verstehe es nicht ... ist es nicht ein Denkfehler? Ich ziehe von meinem täglichen Return der Aktie
            die annualized returns der 10-Jahres-Bonds ab ... ist das nicht Quatsch???
        """)
    print(excess_returns_df)
    print(input('Stop here...'))

    # Step 5: Allocate portfolio based on 200-day moving average
    moving_averages_df = prices_df.rolling(window=200).mean().shift(1)
    print(f"""
    
    moving averages df:
    {moving_averages_df}
    
    
    """)
    portfolio_weights_df = np.where(prices_df > moving_averages_df, 0.2, 0)

    print(f"""
        Portfolio weights:
        Wo prices_df ist größer als mavg yield 0.5 ansonsten 0 - die % hängen von der Anzahl der Aktien ab
        Wir haben Allokationen von 2x .5, 1x plus risk free interest oder alles in risk free interest
        {portfolio_weights_df},
        {type(portfolio_weights_df)}
    """)

    portfolio_weights_df = pd.DataFrame(portfolio_weights_df)
    print(type(portfolio_weights_df))
    print(input('Stop...'))
    portfolio_weights_df['Cash'] = 1 - portfolio_weights_df.sum(axis=1)

    print(f"""
        Neuer Portfolio_Weights_DF inkl. Spalte Cash:
        {portfolio_weights_df}
    """
    )


    portfolio_returns_df = (excess_returns_df * portfolio_weights_df).sum(axis=1)

    # Step 6: Compute mean and standard deviation of the portfolio
    portfolio_mean = portfolio_returns_df.mean()
    portfolio_std = portfolio_returns_df.std()

    print(f"Mean of the portfolio: {portfolio_mean:.6f}")
    print(f"Standard deviation of the portfolio: {portfolio_std:.6f}")

    # Step 7: Compute mean and standard deviation of equally-weighted portfolio
    equal_weights_df = pd.DataFrame(np.ones((len(prices_df), len(symbols))) / len(symbols), index=prices_df.index,
                                    columns=symbols)
    equal_returns_df = (excess_returns_df * equal_weights_df).sum(axis=1)
    equal_mean = equal_returns_df.mean()
    equal_std = equal_returns_df.std()

    print(f"\nMean of equally-weighted portfolio: {equal_mean:.6f}")
    print(f"Standard deviation of equally-weighted portfolio: {equal_std:.6f}")

    # Step 8: Plot cumulative performance of both strategies
    portfolio_cumulative_returns = (1 + portfolio_returns_df).cumprod()
    equal_cumulative_returns = (1 + equal_returns_df).cumprod()

    plt.plot(portfolio_cumulative_returns, label='Portfolio')
    plt.plot(equal_cumulative_returns, label='Equal weights')
    plt.legend()
    plt.title('Cumulative Returns of Portfolio Strategies')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()


def layout():
    """
    Use functions to get reusable code:

    Page layouts must be defined with a variable or function called layout.
    When creating an app with Pages, only use app.layout in your main app.py file.
    :return:
    """

    image_path = {'spx_tnx': 'assets/spx_vs_tnx_nov23.PNG'}

    layout_portfolio = dbc.Container([
                     dbc.Row([
                        dbc.Col([], width=1),
                        dbc.Col([
                            html.H3(children='Portfolio Allocation',
                                    style={'textAlign': 'center'}),
                            html.Br(),
                            html.Hr(),
                            html.Img(src=image_path['spx_tnx'],
                                     style={'box-shadow': '8px 8px 10px 8px dodgerBlue',
                                            'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto',
                                            'width': '50%'},),
                        ], width=10),
                        dbc.Col([], width=1),
                        ]),
                    ])

    return layout_portfolio


if __name__ == '__main__':
    portfolio_allocation()

