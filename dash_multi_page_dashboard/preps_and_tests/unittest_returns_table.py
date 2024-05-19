"""
Before you dive into writing tests, you’ll want to first make a couple of decisions:

        What do you want to test?
        Are you writing a unit test or an integration test?

        Then the structure of a test should loosely follow this workflow:

        Create your inputs
        Execute the code being tested, capturing the output
        Compare the output with an expected result

        For this application, you’re testing sum(). There are many behaviors in sum() you could check, such as:

        Can it sum a list of whole numbers (integers)?
        Can it sum a tuple or set?
        Can it sum a list of floats?
        What happens when you provide it with a bad value, such as a single integer or a string?
        What happens when one of the values is negative?
"""

import os
import inspect
import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np

from investing.p1add_pricedata_to_database import pull_df_from_db

import datetime as dt

from investing.myutils import timestamp

import unittest

# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': 10,
    'min_rows': None,
    'max_rows': 20,
    'precision': 6,
    'float_format': lambda x: f'{x:.4f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)


def normalized_returns_from_database(table: str = 'sp500_adjclose', ticker: str ='all',
                                     start_date='2022-01-03', end_date=dt.datetime.now()):
    """

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

    print("""
           Adjclose several tickers.....
           Todo: muss ich die NaNs nicht noch fillen oder droppen??? Ja. wir droppen bei den returns... below-mentioned
       """, prices_df)



    # Normalize the data:
    """
    Weil "Date" Index ist, kann ich bedenkenlos normalisieren - ID sollte ich droppen, weil es meine range zu
    sehr abweichen lässt...
    """
    prices_df.copy()
    log_prices_df = np.log(prices_df)
    #print(log_prices_df)

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
    #normalized_returns_df.dropna(inplace=True)


    print(f"""
           Normalized daily returns...NaNs got dropped.., only 926 rows/days - starts at 2020-03-20
           {log_returns_df}
       """)

    pctchange_returns_df = (prices_df / prices_df.shift(1))-1
    print('pctchange:', pctchange_returns_df)

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
    # print(min_col)
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


class TestCaseReturnsTable(unittest.TestCase):
    """
    Class for running unittests - Caution: if the test runs, the other functions don't run. Separate them.
    """

    def setUp(self):
        """
        fixture = Inventar
        setUp-routine runs before testing
        :return:
        """
        TEST_INPUT_DIR = '/'
        test_file_name = ''
        try:
            # data = pd.read_csv(TEST_INPUT_DIR + test_file_name, sep=',', header=0)
            data = pull_df_from_db(sql='sp500_adjclose')
        except IOError:
            print('Cannot download from database...')
        self.fixture = data

    def testFiltered_data_from_df(self):
        # Test that the dataframe read in equals what you expect
        # here I need to replace the pull_df-function with a template df I built above-mentioned
        foo = pull_df_from_db(sql='sp500_adjclose')
        assert_frame_equal(self.fixture, foo)

    def test_df_returns_as_expected(self):
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
        assert_frame_equal(df1, df2, check_dtype=False)
        timestamp()


class TestDataFrame(unittest.TestCase):
    def test_dataframe(self):
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3., 4.]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
        self.assertEqual(True, df1.equals(df2))


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(sum([1, 2, 3]), 6, 'Should be 6!')  # add assertion here
        timestamp()


class TestCaseSum(unittest.TestCase):
    def test_list_integers(self):
        """
        Test that it can sum a list of integers.

        :return:
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)
        timestamp()


if __name__ == '__main__':
    normalized_returns_from_database()
    unittest.main()







