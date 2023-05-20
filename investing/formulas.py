import warnings
import datetime as dt

import pandas as pd
import pandas_datareader.data as pdr

from p1add_pricedata_to_database import push_df_to_db_replace, pull_df_from_db

from collections import Counter

from math import factorial
from myutils import get_nth_key, get_nth_value

import yfinance as yf
yf.pdr_override()


def get_daily_returns(symbol: str, startdate=None, enddate=None):
    """

    Wrapper for pandos.io.data.gat_data_yahoo() but overridden by yf.
    Retrieve prices for symbols from yahoo and computes returns based on adjusted closing prices
    Start- and Enddate = None returns historical data from Yahoo since the 70s
    :param symbol: str,
        Symbol name to load, e.g. 'SPY'
    :param startdate: pandas.Timestamp compatible, optional
        Start date of time period to retrieve - STRING!
    :param enddate: pandas.Timestamp compatible, optional
        End date of time period to retrieve - STRING!
    :return: pandas.DataFrame
        Returns of symbol in requested period.
    """

    try:
        df = pdr.get_data_yahoo(symbol, start=startdate, end=enddate)
        returns = df[['Adj Close']].pct_change().dropna()

    except Exception as e:
        warnings.warn(
            'Yahoo Finance read failed: {}, falling back to YFinance'.format(e),
            UserWarning)
        df = yf.download(symbol, start=startdate, end=enddate)
        returns = df[['Adj Close']].pct_change().dropna()

    returns.index = returns.index.tz_localize("UTC")
    returns.columns = [symbol]

    return returns


def normalize_data(df):
    """
    df on input should contain only one column with the price data (plus dataframe index)
    :param df:
    :return: y will be a new column in a dataframe - we will call it 'norm' like so:
    df['norm'] = normalize_data(df['Adj Close'])
    """
    min = df.min()
    max = df.max()
    x = df
    # time series normalization part
    # y will be a column in a dataframe - we will call it 'norm'
    y = (x - min) / (max - min)
    return y


def sma_100(df):
    """
    100days moving average column hinzufügen
    min_periods=0, weil es für die ersten 100 Tage keine Daten gibt
    rolling-function comes inherited from pandas:
    DataFrame.rolling(window, min_periods=None, center=False, win_type=None,
    on=None, axis=0, closed=None, step=None, method='single')

    :param df: refers to a pricedata-df like OHLC
    :return:
    Todo: Funktionen sma20, 50 und 100 dürfen keine df returnen, in denen schon die Spalten droppen
    Todo: Zum Plotten brauche ich einen DF mit AOHLCV und ma20, 50, 100
    """
    df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
    df_100 = df['100ma']
    return df_100


def sma_50(df):
    """

    :param df: refers to a pricedata-df like OHLC
    :return:
    """
    # Todo: ich muss hier schon die extra Spalten droppen
    df['50ma'] = df['Adj Close'].rolling(window=50, min_periods=0).mean()
    df_50 = df['50ma']
    return df_50


def sma_20(df):
    """

    :param df: refers to a pricedata-df like OHLC
    :return:
    """
    df['20ma'] = df['Adj Close'].rolling(window=20, min_periods=0).mean()
    df_20 = df['20ma']
    return df_20


def bayes_law(ticker_event_a: str, ticker_event_b: str, days: int, *args, **kwargs):
    """
    Note that A_AND_B is equivalent to B_AND_A
    Intuition: What's the probability that stock A rises in 100 days, whats the prob. that the SPX rises in 100 days,
    and whats P(A|B)?

    Todo: Das ist nur der Prototyp - ich muss natürlich mit pull_from_db und Schleife alle Aktien verknüpfen!
    Conditional probability question:  "What is the probability of AMZN stock price falling given that
    the SPY index fell earlier?"

    The conditional probability of A given that B has happened can be expressed as:

    If A is: "AMZN price falls" then P(AMZN) is the probability that AMZN falls; and B is: "SPY is already down," and
    P(SPY) is the probability that the SPY fell; then the conditional probability expression reads as "the probability
    that AMZN drops given a SPY decline is equal to the probability that AMZN price declines and SPY declines over the
    probability of a decrease in the SPY index.

    P(AMZN|SPY) = P(AMZN and SPY) / P(SPY)

    P(AMZN and SPY) is the probability of both A and B occurring. This is also the same as the probability of A
    occurring multiplied by the probability that B occurs given that A occurs, expressed as P(AMZN) x P(SPY|AMZN).
    The fact that these two expressions are equal leads to Bayes' theorem, which is written as:

    if, P(AMZN and SPY) = P(AMZN) x P(SPY|AMZN) = P(SPY) x P(AMZN|SPY)

    then, P(AMZN|SPY) = [P(AMZN) x P(SPY|AMZN)] / P(SPY).

    Where P(AMZN) and P(SPY) are the probabilities of Amazon and the SP500 falling, without regard to each other.

    The formula explains the relationship between the probability of the hypothesis before seeing the evidence that
    P(AMZN), and the probability of the hypothesis after getting the evidence P(AMZN|SPY), given a hypothesis
    for Amazon given evidence in the SP500.

    :param ticker_event_b:
    :param ticker_event_a:
    :param days: choose last x days from today to obtain the relationship
    :return: bayes
    """
    main_df = pd.DataFrame()
    probabilities = dict()

    tickers = [ticker_event_a, ticker_event_b]

    # usecase of *args:
    """
    tickers_optional = args
    print(f'optional tickers: {tickers_optional}')
    for i in tickers_optional:
        print(i)
        for ichild in i: print(ichild)
    stopp = input('...')
    

    # usecase of **kwargs: type of output will be a dictionary with the keyword as key :-)
    tickers_optional = kwargs
    print(tickers_optional)
    stopp = input('...')
    """

    for ticker in tickers:
        try:
            df = yf.download(ticker, start='2022-01-01', end=dt.datetime.today())
        except Exception as e:
            warnings.warn(
                f'Yahoo Finance read failed: {e}, falling back to YFinance',
                UserWarning)
            pass

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        df['daily_returns_' + ticker] = df[[ticker]].pct_change().dropna()
        df = df[-days:]
        vals = df['daily_returns_' + ticker].values.tolist()
        print(vals)
        count = 0
        for i in vals:
            if i > 0:
                count += 1
            else:
                continue

        # calculate the probability:
        p_ticker = float(count / days)

        probabilities['p_' + ticker] = p_ticker
        print(probabilities)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    print(tickers)

    # p(A|B) = Intersection / p(B)
    # p(B|A) = Intersection / p(A)
    # Intersection = p(A) + p(B) - Union(A,B)
    # Example last 100 days:
    # Intersection = 0.57 + 0.45 - Union (A OR B - hence all days either A or B rose)
    # Intersection = P(DE AND SPY) - hence all days A AND B rose simultaneously
    # Union_A_OR_B = 0.57 + 0.45 - p_A_AND_B

    # from here use indexing by numbers - some functions for indexing dicts by number, you find in myutils:
    p_A = get_nth_value(probabilities, 0)
    p_B = get_nth_value(probabilities, 1)
    print(f'P(A): {p_A}, P(B): {p_B}')

    # Boolean indexing: any() means OR aka Union, all() means AND aka Intersection, sum just counts the length of array
    union_A_OR_B = (main_df[['daily_returns_' + ticker_event_a, 'daily_returns_' + ticker_event_b]].values > 0).any(axis=1).sum()
    intersection_A_AND_B = (main_df[['daily_returns_' + ticker_event_a, 'daily_returns_' + ticker_event_b]].values > 0).all(axis=1).sum()
    print(f'Union von A und B: {union_A_OR_B}, Intersection von A und B: {intersection_A_AND_B}')
    print(f'P of Union von A und B: {union_A_OR_B/days}, P of Intersection von A und B: {intersection_A_AND_B/days}')

    p_A_IF_B = (intersection_A_AND_B/days) / p_B
    p_B_IF_A = (intersection_A_AND_B/days) / p_A
    print(round(p_A_IF_B, 3), round(p_B_IF_A, 3))

    # Additive law:
    p_union_A_OR_B = p_A + p_B - (intersection_A_AND_B/days)  # you can interchange union and intersection

    # Multiplication rule for case p_B_IF_A:
    intersection_A_AND_B = p_B_IF_A * p_A

    # replace intersection in conditional prob. formula:
    bayes_A_IF_B = p_B_IF_A * p_A / p_B
    print(f"Bayes' Law for p_A_IF_B is: {round(bayes_A_IF_B, 3)}")

    return main_df


def binomial_distribution(area: str = 'sp500', ticker: str = 'DE', days_longterm_prob: int = 100, days_shortterm_prob: int = 5,
                          interval: int = 3, *args, **kwargs):
    """
    Usecases:
    1. Count the rising days of all tickers - that's the success-outcome of the trial (stock rises or falls),
    calculate probability and infer now the probability of 3 rising days within one workweek (5 days)
    2. Todo: Will ticker-companies beat or miss earnings estimates given the historical data and probabilities
    :param area:
    :param interval:
    :param days_shortterm_prob:
    :param days_longterm_prob:
    :param ticker:
    :param args:
    :param kwargs:
    :return:
    """

    #Todo: hier einen match 'europe, sp500, africa' case - default - function einbauen:

    match area:
        case 'sp500':
            df = pull_df_from_db(sql='sp500_adjclose')
        case 'europe':
            df = pull_df_from_db(sql='eurostoxx50_adjclose')
        case 'africa':
            df = pull_df_from_db(sql='africatop250_adjclose')
        case default:
            df = pull_df_from_db(sql='sp500_adjclose')

    print('*'*40)
    print('Beispiele für Binomialverteilung:')
    print('*'*40)
    #print(df)
    print(ticker)
    # stopp = input('stopp...')
    df = df.copy()  # for defragmentation
    df['daily_returns_' + ticker] = df[ticker].pct_change().dropna()

    vals_historic_complete = df['daily_returns_' + ticker].values.tolist()
    vals_historic_complete = vals_historic_complete[1:]
    print(vals_historic_complete)
    count = 0
    for i in vals_historic_complete:
        if i > 0:
            count += 1
        else:
            continue

    # calculate the probability historic:
    p_ticker_historic = float(count / len(vals_historic_complete))

    # Example for indexing rows and columns: dfd.iloc[[0, 2], dfd.columns.get_loc('A')]
    df = df[-days_longterm_prob:]
    vals_longterm = df['daily_returns_' + ticker].values.tolist()
    print(vals_longterm)
    count_longterm = 0
    for i in vals_longterm:
        if i > 0:
            count_longterm += 1
        else:
            continue

    # calculate the probability longterm:
    p_ticker_long = float(count_longterm / len(vals_longterm))

    df = df[-days_shortterm_prob:]
    vals_shortterm = df['daily_returns_' + ticker].values.tolist()
    print(vals_shortterm)
    count_shortterm = 0
    for i in vals_shortterm:
        if i > 0:
            count_shortterm += 1
        else:
            continue

    # calculate the probability shortterm:
    p_ticker_short = float(count_shortterm / len(vals_shortterm))

    print(f'Length of vals is {len(vals_historic_complete)}. ')
    print(f'Length of longtermvals is {len(vals_longterm)}. ')
    print(f'Length of shorttermvals is {len(vals_shortterm)}. ')
    print(f'Count of rising is: {count}, Probability of {ticker} rising within {len(vals_historic_complete)} days: {round(p_ticker_historic, 3)}')
    print(f'Count of rising is: {count_longterm}, Probability of {ticker} rising within {days_longterm_prob} days: {round(p_ticker_long, 3)}')
    print(f'Count of rising is: {count_shortterm}, Probability of {ticker} rising within {days_shortterm_prob} days: {round(p_ticker_short, 3)}')

    n = days_shortterm_prob
    y = interval
    p = p_ticker_long
    p_short = p_ticker_short
    stddev = 0.2
    fact_n = factorial(days_shortterm_prob)
    print(fact_n)
    comb = factorial(n)/(factorial(y)*factorial(n-y))
    comb_1 = factorial(n)/(factorial(1)*factorial(n-1))
    comb_2 = factorial(n)/(factorial(2)*factorial(n-2))
    print('# of Combinations y, 1, 2: ',comb, comb_1, comb_2)
    # stopp = input('stopp...')

    p_bi_dist_long = comb * (p**y) * ((1 - p)**(n-y))

    """ Distribution exactly y matches """
    p_bi_dist_short = comb * (p_short ** y) * ((1 - p_short) ** (n - y))

    """ Distribution maximum y matches """
    p_max_y_short_1 = comb_1 * (p_short ** 1) * ((1 - p_short) ** (n - 1))
    p_max_y_short_2 = comb_2 * (p_short ** 2) * ((1 - p_short) ** (n - 2))
    p_max_y_short_3 = comb * (p_short ** y) * ((1 - p_short) ** (n - y))
    p_max_sum = round(p_max_y_short_1 + p_max_y_short_2 + p_max_y_short_3, 2)

    """ Distribution minimum y matches"""

    """ Distribution more than y matches"""
    p_more_than_y = 1 - p_max_sum

    print(f'Probability with long p of {ticker} of {interval} days rising within {days_shortterm_prob} days is: {p_bi_dist_long}.')
    print(f'Probability short of {ticker} of exactly {interval} days rising within {days_shortterm_prob} days is: {p_bi_dist_short}.')
    print(
        f'Probability short of {ticker} of 1 days rising within {days_shortterm_prob} days is: {p_max_y_short_1}.')
    print(
        f'Probability short of {ticker} of 2 days rising within {days_shortterm_prob} days is: {p_max_y_short_2}.')
    print(
        f'Probability short of {ticker} of 3 days rising within {days_shortterm_prob} days is: {p_max_y_short_3}.')
    print('Maximum y matches: ', p_max_sum)
    print('More than y matches: ', p_more_than_y)

    return p_bi_dist_long, stddev


def option_pricing(strike_price: float, p_up: float, p_down: float,  market_price_max: float = 100,
                   market_price_min: float = 90, ticker: str = 'DE', n: int = 10,
                   premium: float = 0):
    """
    Option: an agreement between two parties for the price of a stock or item at a future point in time. It allows
    one of the sides to decide to buy (call the option) or not to buy the underlying asset on day x. The party who
    decides must pay a premium (kind of fee) to the issuer nevertheless if it buys or not. Question:
    how much we are willing to pay to receive that pact?
    :param strike_price:
    :param p_up:
    :param p_down:
    :param market_price_max:
    :param market_price_min:
    :param ticker:
    :param n:
    :param premium:
    :return:
    """

    df = pull_df_from_db(sql='sp500_adjclose')
    df = df.copy()  # for defragmentation
    df['daily_returns_' + ticker] = df[ticker].pct_change().dropna()
    vals_historic_complete = df['daily_returns_' + ticker].values.tolist()
    vals_historic_complete = vals_historic_complete[1:]
    print(vals_historic_complete)

    count = 0
    for i in vals_historic_complete:
        if i > 0:
            count += 1
        else:
            continue

    # calculate the probability historic:
    p_ticker_historic = float(count / len(vals_historic_complete))

    df = df[-7:]
    vals_shortterm = df['daily_returns_' + ticker].values.tolist()
    print(vals_shortterm)
    count_shortterm = 0
    for i in vals_shortterm:
        if i > 0:
            count_shortterm += 1
        else:
            continue

    # calculate the probability shortterm:
    p_ticker_short = float(count_shortterm / len(vals_shortterm))

    print(f'Length of vals is {len(vals_historic_complete)}. ')
    print(f'Length of vals short is {len(vals_shortterm)}. ')
    print(f'Count of rising is: {count}, Probability of {ticker} rising within {len(vals_historic_complete)} '
          f'days: {round(p_ticker_historic, 3)}')
    print(f'Count of rising is: {count}, Probability of {ticker} rising within {len(vals_shortterm)} '
          f'days: {round(p_ticker_short, 3)}')
    print(df['DE'])

    call_payoff = (n * market_price_max) - premium - (n * strike_price)
    p_down = 1 - p_ticker_short
    print('p_down:', p_down)

    match premium:
        case 0:
            expected_payoff = round((p_ticker_short * (n * (market_price_max - strike_price))), 3)

        case default:
            expected_payoff = round((p_ticker_short * call_payoff) + (p_down * -premium), 3)

    print(f'Surplus is: {call_payoff}')
    print(f'Surplus after call: {expected_payoff}, extra premium you could pay for matching fair deal: {expected_payoff}')


def confidence_intervals_two_means_sector_analysis():
    """
    try it out - does it make sense?
    Compare the annually means of the performance of two sectors of the SP500. We have different sample sizes.
    We can calculate the population variances of the sectors and the means of the stocks.
    Are the populations normally distributed? That's the big question...
    Are the samples really truly independent?
    Problem: we want to find a 95% CI for the difference between the performances (in %) od the stocks of Energy and
    Industrials.

    Idee für die Arbeit: inventory management - nimm Verkaufsdaten (aus Umsatzzahlen) der letzten zwei Jahre, berechne
    den Durchschnitt für jeden Monat und den Standarderror für die Stichprobe - CI 95% für das Folgejahr
    :return:
    """


def hypothesis_testing_outperform_market_on_few_special_days():
    """
    The idea: it is said that the stock market makes its biggest returns within only 5 days during a year. These 5
    days compound for a positive return of the whole year.
    Find such days by looking diligently and name them.
    State the hypothesis:
    H0: On day x the market return of an index/sector/stock is an average return in line with the prior weeks or year,
        H0: return(x) = average + 0 -> means no change
    H1: return(x) = average + 5%
    Collect the market data: special days in 2023...
    :return:
    """


if __name__ == '__main__':
    # print(get_daily_returns('BAS.DE', startdate='2023-01-01', enddate='2023-02-13'))
    main_df = bayes_law('CMCL', 'GC=F', 100, ['DE', 'IMPUY', 'CMCL', 'AU'], optional_list=['DE', 'IMPUY', 'CMCL', 'AU'])
    # push_df_to_db_replace(main_df, 'bayes_law')
    print(main_df)
    #BAS.DE, MUV2.DE - geht noch nicht, weil ich noch kein table europe_adjclose habe
    binomial_distribution(area='sp500', ticker='DE')
    option_pricing(390, 0.4, 0.6, 420, 380, n=10, premium=100)

