from p1_get_financial_data import get_pricedata_yfinance
import datetime as dt
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr

import yfinance as yf
yf.pdr_override()

style.use('ggplot')

'''
Variablen:
'''
start = '2023-01-01' # input('Startdatum im Format YYYY-MM-DD')
end = dt.datetime.now()
tickers = ['CMCL', 'GC=F', 'SPY', 'DE', 'CVX']
tickers_yf = 'DE' # yfinance nutzt nur einen string - auch bei mehreren Tickern ohne Komma

df = get_pricedata_yfinance('CTRA', start, end) # tickers_yf
print(df)

df['Adj Close'].plot()
plt.show()

df[['High', 'Low']].plot()
plt.show()