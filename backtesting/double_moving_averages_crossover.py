import numpy as np
import pandas as pd
import yfinance as yf

import datetime as dt

from openpyxl import writer
import openpyxl


"""
7 Useful Pandas Display Options You Need to Know
"""
# pd.set_option("display.max.columns", None)  # 50, 100, 999 etc. or None for max
# pd.set_option('display.min.rows', None)
# pd.set_option('display.max.rows', None)     # 50, 100, 999 etc. or None for max
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Suppressing scientific notation like 5.55e+07
# pd.set_option('display.float_format', lambda x: f'{x:.3f}')     # das ist wie round()
# pd.set_option('display.precision', 2)       # only changes how the data is displayed not the underlying value
# pd.set_option('display.float_format',  f'{:,.3f}%')   # shows percent-values as percent
# pd.reset_option('display.max_rows')
# pd.reset_option('all')

# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': None,
    'min_rows': None,
    'max_rows': 10,
    'precision': 2,
    'float_format': lambda x: f'{x:.3f}'
    }

for option, value in settings.items():
    pd.set_option("display.{}".format(option), value)


"""
class BacktestingDoubleMovingAveragesCrossover():
    
    What if we had bought Deere-stocks in the year 2013 (ten years ago from the time of writing this code)
    after finding an entry point based on crossing the SMA200 by the SMA14? Would it have been a good idea?

    But actually we just need the first entry point of the last 10 years for backtesting our stockbuy over
    the whole period
    :return: 
    
    Courtesy: https://medium.com/@Jachowskii/how-to-build-a-backtesting-engine-in-python-using-pandas-bc8e532a9e95

"""
# Define a trading strategy:


def sma(array, period):
    """

    :param array: has to be a pd.DataFrame for that we can access the .rolling()-method
    :param period:
    :return:
    """
    return array.rolling(period).mean()


def crossover(array1, array2):
    """
    Entry rules:
    entry (buy) signal when the shorter-term moving average (14 days) crosses above
    the longer-term moving average (200 days)
    :param array1: shorter-term sma
    :param array2: longer-term sma
    :return: bool True/False
    """
    return array1 > array2


def crossunder(array1, array2):
    """
    Exit rules:
    exit (sell) signal when the shorter-term moving average (14 days) crosses below
    the longer-term moving average (200 days)
    :param array1: shorter-term sma
    :param array2: longer-term sma
    :return: bool True/False
    """
    return array1 < array2


def retrieving_stock_data():
    """

    :return:
    """

    """
        Variables:
    """
    ticker = ['AAPL']
    startdate = '2019-01-01'  # input('Startdatum im Format YYYY-MM-DD') usecase for yfinance
    enddate = dt.datetime.now()

    main_df = pd.DataFrame()

    df = yf.download(ticker, start=startdate, end=enddate)

    # Todo: Adj Close ??? That's the big question...
    sma14 = sma(df['Close'], 14)
    sma200 = sma(df['Close'], 200)

    df['sma14'] = sma14
    df['sma200'] = sma200

    # print(sma14)
    # print(sma200)

    enter_rules = crossover(sma14, sma200)
    exit_rules = crossunder(sma14, sma200)

    print('Returns a True/False-bool for the entry: \n', enter_rules)
    print('Returns a True/False-bool for the exit: \n', exit_rules)

    # we find the first True on 2022-10-21 - one weekday later we enter the market with the first position 1:
    check = enter_rules[enter_rules.index == '2022-10-21']  # '2013-12-09'
    check_sma14 = sma14[sma14.index == '2022-10-21']
    check_sma200 = sma200[sma200.index == '2022-10-21']
    print("""
             we expect to find False on the enter_rules on the 16th of October, 2013, 
             since 68.26 < 70.70, i.e. sma14 < sma200.
             This is the starting point for our backtest:
             Interpretation: the first False means SMA14 is below SMA200 - now we wait with our algo until
             SMA14 crosses above SMA200 and then we buy....
        """)
    print('Enter rules check:', check)
    print('SMA14 check:', check_sma14)
    print('SMA200 check:', check_sma200)

    return df, enter_rules, exit_rules


# Define a market position function:

"""
Here, we’re going to create a function that defines the ongoing trades: 
"""


def marketposition_generator(dataset, enter_rules, exit_rules):
    """
    we will create a switch that:

    turns on if enter_rules is True and exit_rules is False and
    turns off if exit_rules is True.
    :param dataset:
    :param enter_rules:
    :param exit_rules:
    :return:
    """
    dataset['enter_rules'] = enter_rules
    dataset['exit_rules'] = exit_rules

    status = 0
    mp = []

    """
    Python’s zip() function creates an iterator that will aggregate elements from two or more iterables. 
    You can use the resulting iterator to quickly and consistently solve common programming problems, 
    like creating dictionaries. In this tutorial, you’ll discover the logic behind the Python zip() 
    function and how you can use it to solve real-world problems.
    
    Using zip() in Python

    Python’s zip() function is defined as zip(*iterables). The function takes in iterables as arguments 
    and returns an iterator. This iterator generates a series of tuples containing elements from each iterable. 
    zip() can accept any type of iterable, such as files, lists, tuples, dictionaries, sets, and so on.
    Passing n Arguments
    
    If you use zip() with n arguments, then the function will return an iterator that generates tuples of 
    length n. To see this in action, take a look at the following code block:
    
    >>> numbers = [1, 2, 3]
    >>> letters = ['a', 'b', 'c']
    >>> zipped = zip(numbers, letters)
    >>> zipped  # Holds an iterator object
    <zip object at 0x7fa4831153c8>
    >>> type(zipped)
    <class 'zip'>
    >>> list(zipped)
    [(1, 'a'), (2, 'b'), (3, 'c')]

    Here, you use zip(numbers, letters) to create an iterator that produces tuples of the form (x, y). 
    In this case, the x values are taken from numbers and the y values are taken from letters. 
    Notice how the Python zip() function returns an iterator. To retrieve the final list object, 
    you need to use list() to consume the iterator.
    """

    # to put it in general: we map i and j
    """
    status is the switch and mp is an empty list that will be populated with the resulting values of status.

    At this point, we create a for loop with zip that works like a… ye, a zipper, enabling us to do a parallel 
    iteration on both enter_rules and exit_rules simultaneously: it will return a single iterator 
    object with all values finally stored into mp that will be:

    mp= 1 (on) whenever enter_rules is True and exit_rules is False and
    mp= 0 (off) whenever exit_rules is True.
    
    For some reason our Trues and Falses were automatically translated into 1 and 0....????
    """
    for (i, j) in zip(enter_rules, exit_rules):
        # our start-status is 0 - we wait for entering the market:
        if status == 0:
            # True is 1 and -1 !!!!
            # enter is True and exit is not True:
            if i == 1 and j != -1:
                # we enter a.k.a buy
                status = 1
        else:
            # exit is True (why is it -1???)
            if j == -1:
                # we leave the market - sell...:
                status = 0
        # we have now a list for all enters and exits of the last ten years
        mp.append(status)
        # Todo: I think something is going wrong here. Once entered (1) the switch never switches back to exit!
        # Todo: on 2023-04-24 we have a false-true combination - it is supposed to leave the market!

    print(mp)
    print('We see the zip-object of enter and exit rules:', zip(enter_rules, exit_rules))
    zip_object = zip(enter_rules, exit_rules)
    print(list(zip_object))
    print('Number of days:', len(mp))

    # we add to our dataframe:
    dataset['mp'] = mp
    # we shift forward by one day so that we buy or sell the next day
    dataset['mp'] = dataset['mp'].shift(1)
    # we fill the NaNs
    dataset.iloc[0, 2] = 0
    print(dataset.iloc[:, 4:])

    return dataset['mp']


def apply_trading_system(dataset, enter_rules, exit_rules, direction: str='long', order_type: str = 'market',
                        enter_level: str = 'open'):
    """

    :param dataset:
    :param direction:
    :param order_type:
    :param enter_level:
    :param enter_rules:
    :param exit_rules:
    :return:
    """
    COSTS = 5.90
    INSTRUMENT = 1 # 1 for stocks, 2 for ... etc.
    OPERATION_MONEY = 10000
    DIRECTION = "long"
    ORDER_TYPE = "market"
    ENTER_LEVEL = dataset['Open']

    # DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)[source]
    #
    # Apply a function along an axis of the DataFrame.
    # Example: our columns are called A and B, so we can override them:
    # df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)

    # we are adding the rules and the yield of the market_postition function to the DataFrame:
    """
    Note: In the previous note, I told you that everything would have been clear: in the lambda function of exit_rules, 
    all values equal True are assigned to -1 while False values are assigned to 0. 
    Thanks to that, marketposition_generator runs wonderfully.
    Guess the second true-value has to be different from the first one (1) for that there won't be any mix-ups.
    """
    dataset['enter_rules'] = enter_rules.apply(lambda x: 1 if x is True else 0)
    dataset['exit_rules'] = exit_rules.apply(lambda x: -1 if x is True else 0)
    dataset['mp'] = marketposition_generator(df, dataset['enter_rules'], dataset['exit_rules'])

    """
    From line 7 to line 12 we define market orders for stocks:

    In lines 7–9 we define the entry_price: if the previous value of mp was zero and the present value is one, 
    i.e. we received a signal, we open a trade at the open price of the next day;
    In lines 10–12 we define number_of_stocks, that is the amount of shares we buy, 
    as the ratio between the initial capital (10k) and the entry_price;
    """
    if ORDER_TYPE == 'market':
        # we need the numpy & - python "and" is ambiguous:
        # dataset..shift(1) means that we move the whole df one row down so that our eye "spots"
        # the prior line (1 above):
        dataset['entry_price'] = np.where((dataset.mp.shift(1) == 0) & (dataset.mp == 1),
                                          # dataset.Open.shift(1), np.nan)
                                          # isn't shift 0 better since we buy the next day? - yes it is...
                                          dataset.Open, np.nan)

        if INSTRUMENT == 1:
            dataset['number_of_stocks'] = np.where((dataset.mp.shift(1) == 0) &
                                                   (dataset.mp == 1), (OPERATION_MONEY - COSTS) / dataset.Open, np.nan)
            dataset.number_of_stocks = dataset.number_of_stocks.apply(lambda x: round(x, 0))

            dataset['investment'] = np.where((dataset.mp.shift(1) == 0) & (dataset.mp == 1),
                                             ((dataset.number_of_stocks * dataset['entry_price'])
                                             + COSTS), np.nan)

    # we forward propagate the value of the entry_price:
    dataset['entry_price'] = dataset['entry_price'].fillna(method='ffill')

    # we round number_of_stocks at the integer value and forward propagate its value as well
    # we already did it above - we HAVE to do it earlier, otherwise we will run into buy problems since
    # you only can buy integer-stock-numbers
    if INSTRUMENT == 1:
        dataset['number_of_stocks'] = dataset['number_of_stocks'].apply(lambda x: round(x, 0)).fillna(method='ffill')

    # numpy where(): condition, if true fill with 'entry' if false return empty string:
    # we associate the label 'entry' to 'events_in' every time mp moves from 0 to 1
    dataset['events_in'] = np.where((dataset.mp == 1) & (dataset.mp.shift(1) == 0), 'entry', '')

    """
    We define the long trades:
    """

    if DIRECTION == 'long':
        if INSTRUMENT == 1:
            # we compute open_operations, i.e. the profit
            # when we enter, we have a number of stocks - so we calculate for every invested day the possible profit:
            # Todo: decide if we take Close or Adj Close:
            dataset['open_operations'] = (dataset.Close - dataset.entry_price) * dataset.number_of_stocks

            # we adjust the previous computation of open_operations whenever we exit the trade: whenever we receive
            # an exit signal, the trade is closed the day after at the open price. Here, round turn costs are included
            dataset['open_operations'] = np.where((dataset.mp == 1) & (dataset.mp.shift(-1) == 0),
                                                  (dataset.Open.shift(-1) - dataset.entry_price) *
                                                  dataset.number_of_stocks - COSTS, dataset.open_operations)

            #dataset['money_total'] =

    else:
        # we replicate for short trades what was said for long trades:
        # to test short trades you just have to set DIRECTION = ‘short'
        if INSTRUMENT == 1:
            dataset['open_operations'] = (dataset.entry_price - dataset.Close) * dataset.number_of_stocks
            dataset['open_operations'] = np.where((dataset.mp == 1) & (dataset.mp.shift(-1) == 0),
                                                  (dataset.entry_price - dataset.Open.shift(-1)) *
                                                  dataset.number_of_stocks - 2 * COSTS, dataset.open_operations)

    # we assign open_operations equal 0 whenever there is no trade in progress
    dataset['open_operations'] = np.where(dataset.mp == 1, dataset.open_operations, 0)

    # we associate the label 'exit' to 'events_out' every time mp moves from 1 to 0, i.e. we receive an exit signal
    dataset['events_out'] = np.where((dataset.mp == 1) & (dataset.exit_rules == -1), 'exit', '')

    # we associate the value of open_operations to operations only when we’re exiting a trade,
    # otherwise nan: by doing so, it will be very easy to aggregate data
    dataset['operations'] = np.where((dataset.exit_rules == -1) & (dataset.mp == 1),
                            dataset.open_operations, np.nan)

    # we define the equity_line for close operations and in the following line
    # it is defined the equity_line for open operations
    dataset['closed_equity'] = dataset.operations.fillna(0).cumsum()
    dataset['open_equity'] = dataset.closed_equity + dataset.open_operations - dataset.operations.fillna(0)
    dataset['money_total'] = np.where(dataset.mp == 1, OPERATION_MONEY - dataset.investment, 0)
    dataset['money_total'] = dataset.money_total.apply(lambda x: round(x, 2))

    dataset.to_csv('data_export/trading_system_export.csv')

    dataset.to_excel('data_export/trading_system_export.xlsx', engine='openpyxl')

    return dataset


if __name__ == '__main__':
    retrieving_stock_data()
    df, enter_rules, exit_rules = retrieving_stock_data()
    dataset = marketposition_generator(df, enter_rules, exit_rules)
    print(dataset)
    apply_trading_system(df, enter_rules, exit_rules)
    dataset = apply_trading_system(df, enter_rules, exit_rules)
    net_profit = dataset['closed_equity'][-1] - 10000
    print(round(net_profit, 2))
    print(type(dataset))
    print(dataset.info(verbose=True))

