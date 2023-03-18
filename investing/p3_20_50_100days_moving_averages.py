import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr

from p1_get_financial_data import get_pricedata_yfinance

from formulas import sma_20, sma_50, sma_100

style.use('ggplot')
#style.use('fivethirtyeight')

'''
Variablen:
'''
start = '2022-01-01' # input('Startdatum im Format YYYY-MM-DD')
end = dt.datetime.now()
tickers = ['CMCL', 'GC=F', 'SPY', 'DE', 'CVX']
tickers_yf = 'DE' # yfinance nutzt nur einen string - auch bei mehreren Tickern ohne Komma
df = get_pricedata_yfinance(input('Ein Ticker in Versalien: '), start, end)


def div_moving_averages(df):
    """

    :return:
    """
    # joined_df_list = []
    start = '2022-01-01'  # input('Startdatum im Format YYYY-MM-DD')
    end = dt.datetime.now()
    print(df)
    df['20ma'] = df['Adj Close'].rolling(window=20, min_periods=0).mean()
    df['50ma'] = df['Adj Close'].rolling(window=50, min_periods=0).mean()
    df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
    print(df)

    return df


if __name__ == '__main__':

    df = div_moving_averages(df)
    df[['Adj Close', '100ma']].plot()
    plt.legend(['Adj Close', '100ma'], loc='upper left')
    plt.show()

    df[['Adj Close', '20ma', '50ma', '100ma']].plot()
    plt.legend(['Adj Close', '20ma', '50ma', '100ma'], loc='upper left')
    plt.show()

    '''
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1) # ax2 teilt sich x-Achse mit ax1

    ax1.plot(df.index, df['Adj Close'])
    ax1.plot(df100.index, df100['100ma'])
    ax1.plot(df20.index, df20['20ma'])
    ax1.plot(df50.index, df50['50ma'])
    # plt.legend('100ma', loc='upper left')???????
    ax2.bar(df.index, df['Volume'])

    plt.show()

    '''
