import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr

from p1_get_financial_data import get_pricedata_yfinance

style.use('ggplot')

'''
Variablen:
'''
start = '2022-01-01' # input('Startdatum im Format YYYY-MM-DD')
end = dt.datetime.now()
tickers = ['CMCL', 'GC=F', 'SPY', 'DE', 'CVX']
tickers_yf = 'DE' # yfinance nutzt nur einen string - auch bei mehreren Tickern ohne Komma

df = get_pricedata_yfinance('AAPL', start, end)

# 100days moving average column hinzufügen
# min_periods=0, weil es für die ersten 100 Tage keine Daten gibt
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
print(df)

df[['Adj Close', '100ma']].plot()
plt.show()

ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1) # ax2 teilt sich x-Achse mit ax1

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()

