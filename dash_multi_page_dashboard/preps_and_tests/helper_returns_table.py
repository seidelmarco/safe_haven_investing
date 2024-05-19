import os
import inspect

import pandas as pd
import numpy as np

from investing.p1add_pricedata_to_database import pull_df_from_db

import datetime as dt


def normalized_returns_from_database(table: str = 'sp500_adjclose', ticker: str ='all',
                                     start_date='2022-01-03', end_date=dt.datetime.now()):
    """
    https://www.codecademy.com/article/normalization

    https://datascience.stackexchange.com/questions/40425/is-it-better-to-use-a-minmax-or-a-log-return-normalization-to-predict-stock-pric

    Log returns are symmetric compared to percentage change. log(a/b) = - log(b/a) and this (less skewness), in theory, leads to better results for most models (linear regression, neural networks).
    Neural networks like lstm work better if the values are close to zero, but the difference in normalizations is usually not that big.
    Any returns (log or percentage) are better than raw values because prices change according to previous prices. Their absolute (raw) values have almost negligible influence compared to previous price.

    I would recommend first to convert to log returns and then normalize. If it is daily prices then I would divide the log returns by something like 0.05. Price changes have very heavy tails in a distribution so I would not suggest using minmax because then you divide by something like 0.5 (which probably was in great depression) and get all values too close to zero. Dividing by standard deviation should also be good.

    But reality is different from theory, so it is better to benchmark. Maybe percentage changes are better because this is the number people see and react to. And markets are a lot about psychology.

    And be prepared to see very high errors and bad models. Financial markets are badly predictable both in practice and theory. According to economic theory if they were predictable and people are rational and have unlimited credit lines then any possibility of earning additional money compared to the whole market will be closed in milliseconds. Only if you find some way to analyze data that noone is currently using only then will you be able to earn money. Neural networks were discussed in 1990s to predict financial markets. So LSTM is not really new in 2018.

    :param table:
    :param ticker:
    :param start_date:
    :param end_date:
    :return: log_returns_df = 0, pctchange_returns_df = 1, minmax_returns_df = 2
    """
    # Step 1: Download daily prices for the five stocks
    symbols = ['CMCL', 'TDG', 'PCAR', 'TRGP', 'CARR']

    """
    hier von DB pullen...
    """

    if '__file__' not in locals():
        fx = inspect.getframeinfo(inspect.currentframe())[0]
    else:
        fx = __file__

    # this command shows your working dir:
    os_dir = os.path.dirname(os.path.abspath(fx))
    print('OS_DIR:\n', os_dir)

    prices_df = pull_df_from_db(sql=table, dates_as_index=True)

    prices_df = prices_df.reset_index().set_index(['Date', 'ID'])


    print("""
           Adjclose several tickers.....
           Todo: muss ich die NaNs nicht noch fillen oder droppen??? Ja. wir droppen bei den returns... below-mentioned
       """, prices_df)

    #ser_id = prices_df['ID']
    #print(ser_id)
    #prices_df.drop(['ID'], axis=1, inplace=True)

    # Normalize the data:
    """
    Weil "Date" Index ist, kann ich bedenkenlos normalisieren - ID sollte ich droppen, weil es meine range zu
    sehr abweichen lässt...
    """
    prices_df.copy()
    log_prices_df = np.log(prices_df)
    print(log_prices_df)

    # Step 2: Calculate daily returns of the stocks
    """
    Bsp. CMCL und TRGP 14.11. minus 13.11.23
    CMCL: ln(12,21) - ln(11,50) = 0,0599
    TRGP: ln(86,43) - ln(84,91) = 0,0177

    Muss ich ln oder kann ich nicht gleich mit % arbeiten....?
    Nein: wenn ich den aktuellen Tag durch den Vortag teile bekomme ich die % und die sind exakt gleich zur Differenz
    der LN-Gleichungen ;-) 

    """
    log_returns_df = np.log(prices_df) - np.log(prices_df.shift(1))

    # die nächste Zeile bewirkt irgendwas Komisches ... muss ich schon vorher die NaNs droppen?
    # normalized_returns_df.dropna(inplace=True)

    print(f"""
           Normalized daily returns...NaNs got dropped.., only 926 rows/days - starts at 2020-03-20
           {log_returns_df}
       """)

    pctchange_returns_df = (prices_df / prices_df.shift(1)) - 1
    print('pctchange - be aware of different rounding compared to log-prices:', pctchange_returns_df)

    print(prices_df.min().min())
    print(prices_df.max().max())
    """
    Min-Max-Formula:
    min = 0
    max = 1
    everything else is in between
    Testidee= 0,5 muss dann 0,5 sein:
    
    x = 0 - 0.5 / -1
    x = min - any / -max
    
    OR
    x = any - min / (max - min) -> den Abstand unten/min bis Preis geteilt durch die ganze Klammer
    x = 0.5 - 0 
    
    """
    min_col = prices_df.min()
    #print(min_col)
    min_scalar = prices_df.min().min()
    print(min_scalar)
    max_scalar = prices_df.max().max()
    print(max_scalar)

    any_price = 2400
    # x = prices_df['MMM']
    minmax_ratio = (any_price - min_scalar) / (max_scalar - min_scalar)
    minmax_percent = ((any_price - min_scalar) / (max_scalar - min_scalar)) * 100
    print(f'minmax_percent: {minmax_percent}%, minmax_ratio: {minmax_ratio}')

    # x ist eine series/ 1 col:
    x = prices_df['MMM']
    # time series normalization part
    minmax_returns_df = (x - min_scalar) / (max_scalar - min_scalar)
    minmax_returns_df.sort_index(ascending=False, inplace=True)

    print(minmax_returns_df)

    return log_returns_df, pctchange_returns_df, minmax_returns_df


def normalize_pricedata(df_1col) -> float:
    """
    df on input should contain only one column with the price data (plus dataframe index)
    :param df_1col:
    :return: y will be a new column in a dataframe - we will call it 'norm' like so:
    df['norm'] = normalize_data(df['Adj Close'])
    """
    min = df_1col.min()
    max = df_1col.max()
    x = df_1col
    # time series normalization part
    y = (x - min) / (max - min)
    return y


if __name__ == '__main__':
    normalized_returns_from_database()

