# % matplotlib inline
from urllib.request import Request, urlopen
import json
import ssl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import openpyxl
import plotly
import cufflinks
import pandas_datareader as web
from pandas.api.types import CategoricalDtype
import tqdm
# Bsp. für alle Schleifen/Iterables:
# sharpe_ratios, wghts = np.column_stack([sharpe_ratios_and_weights(returns) for x in tqdm(range(n))])
import time
from datetime import datetime
from myutils import timestamp
import hidden

# pd.set_option("display.max.columns", None)

# Import the file with the tickers and clean it:

tickers = pd.read_excel('sp500ratings.xlsx')
tickers.set_index('Symbol', inplace=True)
print(tickers)

# Convert credit ratings into categorical data and order them

'''
Ich hatte die NAs schon in xls gelöscht - man kann es aber auch so machen:
Wichtig ist, dass es keine leere erste Spalte (wie in Numbers) gibt
'''
available_stocks = list()
tickers_available = tickers[tickers.index.str.match(('|'.join(available_stocks)))].dropna()
print('')
print('Tickers available w/o NA:')
print(tickers_available)

# datetime object containing current date and time
timestamp()

# Variable für eigene Sortierung bauen:
cat_sp_letters_order = CategoricalDtype(
    categories=[
        'AAA', 'AA +', 'AA', 'AA-', 'A +', 'A', 'A-', 'BBB +', 'BBB', 'BBB- ', 'BB +', 'BB', 'BB-', 'B +', 'B', 'B-'
    ],
    ordered=True
)

tickers_available.astype(cat_sp_letters_order)

tickers_available.to_excel('available_stocks.xlsx')

to_plot = tickers_available['Rating'].value_counts().sort_values(axis=0, ascending=False)  # axis=0, ascending=False
print(to_plot)

# Create bar graph
to_plot = to_plot.reset_index().set_index('index')
ax = to_plot.plot.bar(figsize=(8, 6))

# Insert bar labels
for i in ax.patches:
    ax.text(i.get_x() + 0.25, i.get_height() + 1.5, str(round((i.get_height()))), fontsize=10, color='dimgrey',
            ha='center')

plt.box(False)
# plt.show()

# Download daily stock price data for S&P500 stocks

# datetime object containing current date and time
timestamp()


def pricedata_per_group(sourcefile):
    """

    :param sourcefile:
    :return:
    """
    group = pd.read_excel(sourcefile)
    tick = list(group['Symbol'])
    price_data = web.get_data_yahoo(tick,
                                    start='2022-10-13',
                                    end='2022-11-12')['Adj Close']

    # Drop na values from price matrix
    price_data_no_na = price_data.dropna(axis=0)

    print('Pricedata no NAs', price_data_no_na)

    # Transform the price matrix into a return matrix (prozentuale tägliche returns)
    '''
    𝑅𝑒𝑡𝑢𝑟𝑛𝑖,𝑇 = 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇 − 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1/𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 = 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇/𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 − 1 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1 𝑆𝑡𝑜𝑐𝑘 𝑃𝑟𝑖𝑐𝑒𝑖,𝑇−1
    '''
    # Let’s use the dataframe.shift() function to shift the column axis by 1 period in a positive direction
    returns = (price_data_no_na / price_data_no_na.shift(1) - 1)[
              1:]  # das Startdatum wegslicen, wir würden dort ohne 1: NaN sehen
    # die Shift-Formel hat den 1.11.durch den 31.10 geteilt und einen Verlust von 0,3895% ergeben
    # die -1 nach der Shift-Klammer zieht einfach nur 1 ab
    # shift(1) scheint eine Zeile zurückzugehen (schiebt eine Zeile von oben auf den cursor (die aktuelle loop))
    # datetime object containing current date and time
    datetime()
    print(returns)
    return price_data_no_na, returns


'''
returns_group1 = pricedata_per_group('Group_1_AAA_to_A.xlsx')
returns_group1.to_csv('daily_returns_group1_AAAtoA.csv')
returns_group2 = pricedata_per_group('Group_2_A-.xlsx')
returns_group2.to_csv('daily_returns_group2_A-.csv')
print(type(returns_group2))
returns_group3 = pricedata_per_group('Group_3_BBB+.xlsx')
returns_group3.to_csv('daily_returns_group3_BBB+.csv')
returns_group4 = pricedata_per_group('Group_4_BBB.xlsx')
returns_group4.to_csv('daily_returns_group4_BBB.csv')
# in 5 Discovery wieder einfügen, auch water towers suchen, siehe sp500 ratings
returns_group5 = pricedata_per_group('Group_5_BBB-.xlsx')
returns_group5.to_csv('daily_returns_group5_BBB-.csv')
returns_group6 = pricedata_per_group('Group_6_BB+_to_B-.xlsx')
returns_group6.to_csv('daily_returns_group6_BB+B-.csv')
'''

# datetime object containing current date and time
timestamp()


# Funktion: Produce random weights for portfolio stocks, constraining them to be additive to one
# produce x random weight which adds up to one (where x is the number of stocks in the group)


def weights(num_stocks):
    k = np.random.rand(num_stocks)
    return k / sum(k)


# print(weights(140))

# Funktion: one to calculate and return the standard deviations and return of the portfolio
# Calculate mean daily return and daily stdev of portfolio i


def returns_and_stdevs(varreturns):
    mean_returns_vector = np.asmatrix(np.mean(varreturns, axis=1))
    weights_vector = np.asmatrix(weights(varreturns.shape[0]))
    covariance_matrix = np.asmatrix(np.cov(varreturns))
    return_port = weights_vector * mean_returns_vector.T
    stdev_port = np.sqrt(weights_vector * covariance_matrix * weights_vector.T)
    # print(return_port, stdev_port)
    return return_port, stdev_port


# Create 100,000 portfolios, and record mean daily returns and daily standard deviations for each of them

# n = 100000
# für eine bessere Grafik nur 10000 nehmen
n = 1000
# oben das print rausnehmen, sonst bekomme ich 100000 Zeilen mit return_port und stdev_port
'''
numpy.column_stack
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.column_stack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])
'''

# hier für plot-Test nur auf die schon gelesenen csv zurückgreifen:
returns_group1 = pd.read_csv('daily_returns_group1_AAAtoA.csv')
returns_group1.set_index('Date', inplace=True)
print(returns_group1)
returns_group2 = pd.read_csv('daily_returns_group2_A-.csv')
returns_group2.set_index('Date', inplace=True)
print(returns_group2)
returns_group3 = pd.read_csv('daily_returns_group3_BBB+.csv')
returns_group3.set_index('Date', inplace=True)
print(returns_group3)
returns_group4 = pd.read_csv('daily_returns_group4_BBB.csv')
returns_group4.set_index('Date', inplace=True)
print(returns_group4)
returns_group5 = pd.read_csv('daily_returns_group5_BBB-.csv')
returns_group5.set_index('Date', inplace=True)
print(returns_group5)
returns_group6 = pd.read_csv('daily_returns_group6_BB+B-.csv')
returns_group6.set_index('Date', inplace=True)
print(returns_group6)

print('Hier müsste der Fehler auftauchen...manchmal axis out of bound, manchmal läuft es')
means, stds = np.column_stack([returns_and_stdevs(returns_group2) for x in tqdm.tqdm(range(n))])

# Plot portfolios' mean daily returns and daily standard deviations previously created
fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(stds, means, 'o', markersize=4)
plt.xlabel('Standard Deviation', fontsize=15, alpha=0.8)
plt.ylabel('Mean return', fontsize=15, alpha=0.8)
plt.title('Return and volatility of {} portfolios'.format(str(n)), fontsize=20, alpha=0.8, fontweight='bold')
plt.box(False)
plt.grid()
plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom='on')

# datetime object containing current date and time
timestamp()

'''
The scope of the analysis is to take the portfolio which maximizes the Sharpe ratio. 
Therefore, among the 100,000 simulated portfolios, the portfolio with the maximum Sharpe Ratio is going to be selected, 
and then considered in the comparative analysis, both against the other groups 
(to compare portfolios with different classes of issuer credit risk) and over time 
(to see if the effect is due to a structural component of the market, 
or if Covid-19 changed the physiognomy of the latter).

The Sharpe Ratio is calculated by dividing the portfolio return by its volatility (the standard deviation of the 
portfolio calculated for the same time frame as for the portfolio returns):

𝑆h𝑎𝑟𝑝e𝑅𝑎𝑡𝑖𝑜 𝑇 = 𝑃𝑜𝑟𝑡𝑓𝑜𝑙𝑖𝑜𝑅𝑒𝑡𝑢𝑟𝑛 𝑇 / 𝑃𝑜𝑟𝑡𝑓𝑜𝑙𝑖𝑜 𝑆𝑡𝑎𝑛𝑑𝑎𝑟𝑑 𝐷𝑒𝑣𝑖𝑎𝑡𝑖𝑜𝑛 𝑇
'''


# Function that implements the same calculation explained above, with the exception that it returns
# Sharpe Ratio and weights instead of return and standard deviation


def sharpe_ratios_and_weights(varreturns):
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
    return_port = weights_vector * mean_returns_vector.T  # NumPy: Transpose ndarray (swap rows and columns, rearrange axes)
    stdev_port = np.sqrt(weights_vector * covariance_matrix * weights_vector.T)
    return float(return_port / stdev_port), weights_vector


# Simulate 100,000 portfolios and store the Sharpe Ratios and weights for each of them
# n = 100000
n = 1000

sharpe_ratios1, wghts1 = np.column_stack(
    [sharpe_ratios_and_weights(returns_group1) for x in tqdm.tqdm(range(n), colour='green')])
sharpe_ratios2, wghts2 = np.column_stack(
    [sharpe_ratios_and_weights(returns_group2) for x in tqdm.tqdm(range(n), colour='green')])
sharpe_ratios3, wghts3 = np.column_stack(
    [sharpe_ratios_and_weights(returns_group3) for x in tqdm.tqdm(range(n), colour='green')])
sharpe_ratios4, wghts4 = np.column_stack(
    [sharpe_ratios_and_weights(returns_group4) for x in tqdm.tqdm(range(n), colour='green')])
sharpe_ratios5, wghts5 = np.column_stack(
    [sharpe_ratios_and_weights(returns_group5) for x in tqdm.tqdm(range(n), colour='green')])
sharpe_ratios6, wghts6 = np.column_stack(
    [sharpe_ratios_and_weights(returns_group6) for x in tqdm.tqdm(range(n), colour='green')])

# Find the maximum sharpe ratio
print('')
print('Interpretation: Generally, the higher the Sharpe ratio, the more attractive the risk-adjusted return.')
print('')
# print('Größte sharpe-ratio 1:', max(sharpe_ratios1))
# print(type(sharpe_ratios1))
print('Größte sharpe-ratio 2:', max(sharpe_ratios2))
# print('Größte sharpe-ratio 3:', max(sharpe_ratios3))
# print('Größte sharpe-ratio 4:', max(sharpe_ratios4))
# print('Größte sharpe-ratio 5:', max(sharpe_ratios5))
# print('Größte sharpe-ratio 6:', max(sharpe_ratios6))

'''
df_max_sharpes = pd.DataFrame({'AAA': max(sharpe_ratios1), 'A-': max(sharpe_ratios2),
                               'BBB+': max(sharpe_ratios3), 'BBB': max(sharpe_ratios4),
                               'BBB-': max(sharpe_ratios5), 'B-': max(sharpe_ratios6)}, index=['Max'])
'''
df_max_sharpes = pd.DataFrame({'Max Sharpes by Group': [max(sharpe_ratios1), max(sharpe_ratios2),
                                                        max(sharpe_ratios3), max(sharpe_ratios4),
                                                        max(sharpe_ratios5), max(sharpe_ratios6)]}, index=['AAA', 'A-'
    , 'BBB+', 'BBB',
                                                                                                           'BBB-',
                                                                                                           'B-'],
                              dtype=None)

# convert columns to numeric - brauche ich nicht mehr, weil ich oben den df_max_sharpes besser aufgebaut habe:
# df_max_sharpes['AAA']=df_max_sharpes['AAA'].astype(float)
# df_max_sharpes['A-']=df_max_sharpes['A-'].astype(float)
# df_max_sharpes['BBB+']=df_max_sharpes['BBB+'].astype(float)
# df_max_sharpes['BBB']=df_max_sharpes['BBB'].astype(float)
# df_max_sharpes['BBB-']=df_max_sharpes['BBB-'].astype(float)
# df_max_sharpes['B-']=df_max_sharpes['B-'].astype(float)

# df_max_sharpes.set_index(['AAA'], inplace=True)
print(df_max_sharpes)
print(df_max_sharpes.columns)

# Create bar graph

groups = df_max_sharpes.sort_values(by=['Max Sharpes by Group'], axis=0, ascending=False)
ax = groups.plot(kind='bar', figsize=(8, 6), fontsize=8, rot=4)
plt.grid(visible=True, axis='y')
plt.xlabel('Issuer Credit Groups', fontsize=15, alpha=0.8)
plt.ylabel('Sharpe Ratio', fontsize=15, alpha=0.8)
plt.title('Sharpe Ratio of {} portfolios per Group'.format(str(n)), fontsize=20, alpha=0.8, fontweight='bold')

# Insert bar labels
for i in ax.patches:
    ax.annotate(str(round(i.get_height(), 3)), (i.get_x() * 1.005, i.get_height() * 1.005))
    # ax.text(i.get_x() + 0.25, i.get_height() + 1.5, str(round((i.get_height()))), fontsize=10, color='dimgrey', ha='center')

# plt.box(False)
# plt.show()

'''
Ich nehme die 6 Werte von den max sharpes oben und bastle diese als floats in einen neuen DF mit 
allen Row- und Column-Bezeichnungen separat
pd.DataFrame()

siehe Screenshot im Handy und Tutorial 538-Theme plot :-)
einfach dieses Tutorial mit den 6 Werten nachbauen

max_sharpes = [2.506863,  2.419919,  2.173488,  1.614353,  2.525957,  2.155645]
dia = pd.Series(max_sharpes, index=range(len(max_sharpes)))
dia.plot.bar(figsize=(8, 6), color='blue')
'''

# datetime object containing current date and time
timestamp()

deadend = input('Press Enter for showing the plots: ')
plt.show()
deadend2 = input('Press Enter for showing the next plots: ')

'''
Comparative analysis by group based on Issuer Credit Rating over time
'''
# Download daily stock price data for the 6 portfolios

'''
group_1 = pd.read_excel("Group_1_AAA_to_A.xlsx")
tick = list(group_1['Symbol'])
price_data_1 = web.get_data_yahoo(tick,
                                start = '2020-01-01',
                                end = '2020-03-23')['Adj Close']
print(price_data_1)


group_2 = pd.read_excel("Group_2_A-.xlsx")
tick = list(group_2['Symbol'])
price_data_2 = web.get_data_yahoo(tick,
                                start = '2020-01-01',
                                end = '2020-03-23')['Adj Close']
print(price_data_2)

group_3 = pd.read_excel("Group_3_BBB+.xlsx")
tick = list(group_3['Symbol'])
price_data_3 = web.get_data_yahoo(tick,
                                start = '2020-01-01',
                                end = '2020-03-23')['Adj Close']
print(price_data_3)

group_4 = pd.read_excel("Group_4_BBB.xlsx")
tick = list(group_4['Symbol'])
price_data_4 = web.get_data_yahoo(tick,
                                start = '2020-01-01',
                                end = '2020-03-23')['Adj Close']
print(price_data_4)

group_5 = pd.read_excel("Group_5_BBB-.xlsx")
tick = list(group_5['Symbol'])
price_data_5 = web.get_data_yahoo(tick,
                                start = '2020-01-01',
                                end = '2020-03-23')['Adj Close']
print(price_data_5)

group_6 = pd.read_excel("Group_6_BB+_to_B-.xlsx")
tick = list(group_6['Symbol'])
price_data_6 = web.get_data_yahoo(tick,
                                start = '2020-01-01',
                                end = '2020-03-23')['Adj Close']
print(price_data_6)


#2019
group_1 = pd.read_excel("Group_1_AAA_to_A.xlsx")
tick = list(group_1['Symbol'])
price_data_119 = web.get_data_yahoo(tick,
                                start='2019-01-01',
                                end='2019-12-31')['Adj Close']
print(price_data_119)


group_2 = pd.read_excel("Group_2_A-.xlsx")
tick = list(group_2['Symbol'])
price_data_219 = web.get_data_yahoo(tick,
                                start='2019-01-01',
                                end='2019-12-31')['Adj Close']
print(price_data_219)

group_3 = pd.read_excel("Group_3_BBB+.xlsx")
tick = list(group_3['Symbol'])
price_data_319 = web.get_data_yahoo(tick,
                                start='2019-01-01',
                                end='2019-12-31')['Adj Close']
print(price_data_319)

group_4 = pd.read_excel("Group_4_BBB.xlsx")
tick = list(group_4['Symbol'])
price_data_419 = web.get_data_yahoo(tick,
                                start='2019-01-01',
                                end='2019-12-31')['Adj Close']
print(price_data_419)

group_5 = pd.read_excel("Group_5_BBB-.xlsx")
tick = list(group_5['Symbol'])
price_data_519 = web.get_data_yahoo(tick,
                                start='2019-01-01',
                                end='2019-12-31')['Adj Close']
print(price_data_519)

group_6 = pd.read_excel("Group_6_BB+_to_B-.xlsx")
tick = list(group_6['Symbol'])
price_data_619 = web.get_data_yahoo(tick,
                                start='2019-01-01',
                                end='2019-12-31')['Adj Close']
print(price_data_619)


price_data_1.to_csv('pricedata1_2020.csv')
price_data_2.to_csv('pricedata2_2020.csv')
price_data_3.to_csv('pricedata3_2020.csv')
price_data_4.to_csv('pricedata4_2020.csv')
price_data_5.to_csv('pricedata5_2020.csv')
price_data_6.to_csv('pricedata6_2020.csv')

price_data_119.to_csv('pricedata1_2019.csv')
price_data_219.to_csv('pricedata2_2019.csv')
price_data_319.to_csv('pricedata3_2019.csv')
price_data_419.to_csv('pricedata4_2019.csv')
price_data_519.to_csv('pricedata5_2019.csv')
price_data_619.to_csv('pricedata6_2019.csv')
'''

# For each group, calculate the period return of every stock in the group and then store the median in a variable
price_data_1 = pd.read_csv('pricedata1_2020.csv')
first_last_1 = price_data_1.iloc[[0, 55]]
# print(first_last_1)
# print(first_last_1.columns)
first_last_1 = first_last_1[first_last_1.columns.drop('Date')].astype(float)
returns_1 = first_last_1.pct_change()
# print(returns_1)
median_loss_1 = returns_1.iloc[1].dropna().median()

price_data_2 = pd.read_csv('pricedata2_2020.csv')
first_last_2 = price_data_2.iloc[[0, 55]]
first_last_2 = first_last_2[first_last_2.columns.drop('Date')].astype(float)
returns_2 = first_last_2.pct_change()
median_loss_2 = returns_2.iloc[1].dropna().median()

price_data_3 = pd.read_csv('pricedata3_2020.csv')
first_last_3 = price_data_3.iloc[[0, 55]]
first_last_3 = first_last_3[first_last_3.columns.drop('Date')].astype(float)
returns_3 = first_last_3.pct_change()
median_loss_3 = returns_3.iloc[1].dropna().median()

price_data_4 = pd.read_csv('pricedata4_2020.csv')
first_last_4 = price_data_4.iloc[[0, 55]]
first_last_4 = first_last_4[first_last_4.columns.drop('Date')].astype(float)
returns_4 = first_last_4.pct_change()
median_loss_4 = returns_4.iloc[1].dropna().median()

price_data_5 = pd.read_csv('pricedata5_2020.csv')
first_last_5 = price_data_5.iloc[[0, 55]]
first_last_5 = first_last_5[first_last_5.columns.drop('Date')].astype(float)
returns_5 = first_last_5.pct_change()
median_loss_5 = returns_5.iloc[1].dropna().median()

price_data_6 = pd.read_csv('pricedata6_2020.csv')
first_last_6 = price_data_6.iloc[[0, 55]]
first_last_6 = first_last_6[first_last_6.columns.drop('Date')].astype(float)
returns_6 = first_last_6.pct_change()
median_loss_6 = returns_6.iloc[1].dropna().median()

# 2019
price_data_119 = pd.read_csv('pricedata1_2019.csv')
first_last_119 = price_data_119.iloc[[0, 251]]
first_last_119 = first_last_119[first_last_119.columns.drop('Date')].astype(float)
returns_1 = first_last_119.pct_change()
median_win_119 = returns_1.iloc[1].dropna().median()

price_data_219 = pd.read_csv('pricedata2_2019.csv')
first_last_219 = price_data_219.iloc[[0, 251]]
first_last_219 = first_last_219[first_last_219.columns.drop('Date')].astype(float)
returns_2 = first_last_219.pct_change()
median_win_219 = returns_2.iloc[1].dropna().median()

price_data_319 = pd.read_csv('pricedata3_2019.csv')
first_last_319 = price_data_319.iloc[[0, 251]]
first_last_319 = first_last_319[first_last_319.columns.drop('Date')].astype(float)
returns_3 = first_last_319.pct_change()
median_win_319 = returns_3.iloc[1].dropna().median()

price_data_419 = pd.read_csv('pricedata4_2019.csv')
first_last_419 = price_data_419.iloc[[0, 251]]
first_last_419 = first_last_419[first_last_419.columns.drop('Date')].astype(float)
returns_4 = first_last_419.pct_change()
median_win_419 = returns_4.iloc[1].dropna().median()

price_data_519 = pd.read_csv('pricedata5_2019.csv')
first_last_519 = price_data_519.iloc[[0, 251]]
first_last_519 = first_last_519[first_last_519.columns.drop('Date')].astype(float)
returns_5 = first_last_519.pct_change()
median_win_519 = returns_5.iloc[1].dropna().median()

price_data_619 = pd.read_csv('pricedata6_2019.csv')
first_last_619 = price_data_619.iloc[[0, 251]]
first_last_619 = first_last_619[first_last_619.columns.drop('Date')].astype(float)
returns_6 = first_last_619.pct_change()
median_win_619 = returns_6.iloc[1].dropna().median()

print('Losses 20', median_loss_1, median_loss_2, median_loss_3, median_loss_4, median_loss_5, median_loss_6)
print('Wins 19', median_win_119, median_win_219, median_win_319, median_win_419, median_win_519, median_win_619)

Losses_20 = [median_loss_1, median_loss_2, median_loss_3, median_loss_4, median_loss_5, median_loss_6]
Wins_19 = [median_win_119, median_win_219, median_win_319, median_win_419, median_win_519, median_win_619]
# groups = pd.Categorical(['AAA to A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+ to B-'], ordered=True)
data = {
    'Losses 20': pd.Series(Losses_20, index=['AAA to A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+ to B-']),
    'Wins 19': pd.Series(Wins_19, index=['AAA to A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+ to B-'])
}
print(data)
df = pd.DataFrame(data)
# df.style - nur in Jupiter Notebooks
print(df)
df.plot.barh(figsize=(9, 6))
plt.show()
print(df.index, df.columns)


def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    # Ignore SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    response = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    dataframe = urlopen(response, context=ctx)
    # html_doc = document.read()
    # response = urlopen(url, cafile=certifi.where()) #certifi is deprecated - lieber darauf verzichten
    data = dataframe.read().decode("utf-8")
    return json.loads(data)


# Create an empty list and a list of the stocks included in the group
df1 = pd.DataFrame()
stocks = pd.read_excel('Group_4_BBB.xlsx')
tickers = list(stocks['Symbol'])

key = hidden.secrets_fmp()
# For each ticker in the previous list, create a loop to query the data and store it in the previously
# created data frame

'''
for i in tqdm.tqdm(tickers):

    i.upper()

    api_profile_url = 'https://financialmodelingprep.com/api/v3/profile/' + i + '?apikey='+ key
    data_profile = get_jsonparsed_data(api_profile_url)
    data = pd.Series(data_profile[0])
    df1[i] = data

df1.to_csv('Group3_sectors.csv')
df1.to_excel('Group3_sectors.xlsx', startcol=0, startrow=0)
df1 = df1.loc['sector']
print(df1)
'''
# Create a data frame with the number of stock by industry for the group

# df.loc['sector] scheint nicht zu gehen, weil wir zu csv anstatt xls schreiben
# sectors_group1 = pd.read_csv('Group1_sectors.csv')
sectors_group1 = pd.read_excel('Group1_sectors.xlsx')
'''
Wir hatten eine unname-Spalte, die im Plot als Sektor mit Anzahl 1 gezählt wurde. Folgende Lösung:
There are situations when an Unnamed: 0 column in pandas comes when you are reading CSV file . 
The simplest solution would be to read the "Unnamed: 0" column as the index. 
So, what you have to do is to specify an index_col=[0] argument to read_csv() function, 
then it reads in the first column as the index.


index_col=[0]

While you read csv file, if you set index_col=[0] you're explicitly stating to treat the first column as the index.


You can solve this issue by using index_col=0 in you read_csv() function.
index=False


In most cases, it is caused by your to_csv() having been saved along with an "Unnamed: 0" index. 
You could have avoided this mistakes in the first place by using "index=False" if the output CSV was created 
in DataFrame.

You can get ride of all Unnamed columns from your DataFrame by using regex.

http://net-informations.com/ds/err/unnamed.htm

'''
sectors_group1.drop(sectors_group1.filter(regex="Unname"), axis=1, inplace=True)
print('Gruppe 1 Zeilen und Spalten', sectors_group1.index, sectors_group1.columns)
print('DF Gruppe 1', sectors_group1)

sectors_group1 = sectors_group1.iloc[19].value_counts(sort=False)
print('Gruppe 1 nur Sektoren', sectors_group1)

# sectors_group2 = pd.read_csv('Group2_sectors.csv')
sectors_group2 = pd.read_excel('Group2_sectors.xlsx')
sectors_group2.drop(sectors_group2.filter(regex="Unname"), axis=1, inplace=True)
# print(sectors_group2.index, sectors_group2.columns)
# print(sectors_group2)

sectors_group2 = sectors_group2.iloc[19].value_counts(sort=False)
# print(sectors_group2)

# sectors_group3 = pd.read_csv('Group3_sectors.csv')
sectors_group3 = pd.read_excel('Group3_sectors.xlsx')
sectors_group3.drop(sectors_group3.filter(regex="Unname"), axis=1, inplace=True)
# print(sectors_group3.index, sectors_group3.columns)
# print(sectors_group3)

sectors_group3 = sectors_group3.iloc[19].value_counts(sort=False)
# print(sectors_group3)

sectors_group4 = pd.read_csv('Group4_sectors.csv')
sectors_group4.drop(sectors_group4.filter(regex="Unname"), axis=1, inplace=True)
# sectors_group4 = pd.read_excel('Group4_sectors.xlsx')
# print(sectors_group4.index, sectors_group4.columns)
# print(sectors_group4)

sectors_group4 = sectors_group4.iloc[19].value_counts(sort=False)
# print(sectors_group4)

sectors_group5 = pd.read_csv('Group5_sectors.csv')
sectors_group5.drop(sectors_group5.filter(regex="Unname"), axis=1, inplace=True)
# sectors_group5 = pd.read_excel('Group5_sectors.xlsx')
# print(sectors_group5.index, sectors_group5.columns)
# print(sectors_group5)

sectors_group5 = sectors_group5.iloc[19].value_counts(sort=False)
# print(sectors_group5)

sectors_group6 = pd.read_csv('Group6_sectors.csv')
sectors_group6.drop(sectors_group6.filter(regex="Unname"), axis=1, inplace=True)
# sectors_group6 = pd.read_excel('Group6_sectors.xlsx')
# print(sectors_group6.index, sectors_group6.columns)
# print(sectors_group6)

sectors_group6 = sectors_group6.iloc[19].value_counts(sort=False)
# print(sectors_group6)


# Create the graph and apply some formatting

ax = sectors_group6.plot.bar(figsize=(8, 6), fontsize=8, rot=45)
plt.yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# Set individual bar labels using the above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x() + 0.25, i.get_height() + 0.25, str(round((i.get_height()))), fontsize=10, color='dimgrey',
            ha='center')

plt.show()

# Create an empty data frame

sectors_allgroups = pd.DataFrame()

# Standardize industry data by turning it into percent form, while also storing them into a single dataset
sectors_allgroups['AAA to A'] = sectors_group1 / sum(sectors_group1)  # sectors_group1 ist summe mit value_counts
sectors_allgroups['A-'] = sectors_group2 / sum(sectors_group2)
sectors_allgroups['BBB+'] = sectors_group3 / sum(sectors_group3)
sectors_allgroups['BBB'] = sectors_group4 / sum(sectors_group4)
sectors_allgroups['BBB-'] = sectors_group5 / sum(sectors_group5)
sectors_allgroups['BB+ to B-'] = sectors_group6 / sum(sectors_group6)

pd.set_option("display.max.columns", None)

print(sectors_allgroups)

# Export the dataset in excel format
sectors_allgroups.to_excel('Sectorial allocation by group.xlsx')

allsectors_allgroups = pd.read_excel('Sectorial allocation by group.xlsx')
print(allsectors_allgroups.index, allsectors_allgroups.columns)
ax = allsectors_allgroups.plot.bar(figsize=(8, 6), fontsize=8, rot=45)
plt.show()

# Read excel file and store data in a data frame
# oder die csv nehmen

# Figure 12: Box plot of industry allocation of group 2 to 6, with an overlapping scatter plot for group 1

sectors_groups2to6 = allsectors_allgroups.drop(columns=['AAA to A'])
print(sectors_groups2to6)

sectors_groups2to6.to_excel('For box plot.xlsx')
# df = pd.read_excel('For box plot.xlsx')
print(sectors_groups2to6.index)
print(sectors_groups2to6.columns)
sectors_groups2to6.set_index('Unnamed: 0', inplace=True)
sectors_groups2to6 = sectors_groups2to6.T

print(sectors_groups2to6)

ax = sectors_groups2to6[1:].plot.box(figsize=(10, 6), color='midnightblue', rot=45)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
sectors_groups1 = allsectors_allgroups.drop(columns=['A-', 'BBB+', 'BBB', 'BBB-', 'BB+ to B-'])
sectors_groups1.set_index('Unnamed: 0', inplace=True)
sectors_groups1 = sectors_groups1.T
print(sectors_groups1)

count = 0
sum_sectors1 = 0
for row in sectors_group1:
    sum_sectors1 += row

print('Summe Sektoren in Gruppe 1:', sum_sectors1)

for row in sectors_group1:
    print('Jede Reihe aus sectors_group1', row)
    x = count + 1
    count += 1
    y = row / sum_sectors1
    plt.scatter(x, y, color='royalblue', s=30)  # s = marker sizes

plt.show()

print('Gruppe 1 nur Sektoren', sectors_group1)










