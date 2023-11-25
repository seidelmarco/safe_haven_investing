import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': None,
    'min_rows': None,
    'max_rows': 226,
    'precision': 4,
    'float_format': lambda x: f'{x:.2f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)


def data_collector():
    """

    :return:
    """
    raw_data = pd.read_csv('data_csv_excel/fin_indicators_us_stocks/2018_Financial_Data.csv', index_col='Unnamed: 0')
    print(raw_data.head())
    print(raw_data.describe(include='all'))

    data = raw_data.copy(deep=True)
    print(data.columns)

    data.rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True)
    print(data.columns)
    print(data.axes)


    # GENIAL!
    print(data.dtypes)
    print(data.head())


def preprocessing():
    """
    Cleaning and cleansing: hier mit outliers, nan, categoricals etc. arbeiten.
    :return:
    """


def indicators_model():
    """
    Meine Modellidee - alle Features auf 5 Punkte prüfen...

    Check for OLS-Assumptions...

    y_ttm_hat = earnings + revenue + costs + divausschüttungsquote + debts + free cash flow,dcf, um den risiko-
     freien Zins mit reinzubringen etc..
    :return:
    """


if __name__ == '__main__':
    data_collector()
    indicators_model()

