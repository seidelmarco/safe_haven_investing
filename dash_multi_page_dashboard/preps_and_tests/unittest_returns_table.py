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
    # Step 1: Download daily prices for the five stocks
    symbols = ['CMCL', 'TDG', 'PCAR', 'TRGP', 'CARR']
    # quick prototyping:
    # symbols = ['CMCL', 'TRGP']

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

    prices_df = pull_df_from_db(sql=table)

    print("""
           Adjclose several tickers.....
           Todo: muss ich die NaNs nicht noch fillen oder droppen??? Ja. wir droppen bei den returns... below-mentioned
       """, prices_df)

    # Normalize the data:
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
    normalized_returns_df = np.log(prices_df) - np.log(prices_df.shift(1))

    # die nächste Zeile bewirkt irgendwas Komisches ... muss ich schon vorher die NaNs droppen?
    #normalized_returns_df.dropna(inplace=True)


    print(f"""
           Normalized daily returns...NaNs got dropped.., only 926 rows/days - starts at 2020-03-20
           {normalized_returns_df}
       """)

    return normalized_returns_df


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







