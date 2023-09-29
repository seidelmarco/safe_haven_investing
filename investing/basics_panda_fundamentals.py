import pandas as pd
import numpy as np


# use the power of a dictionary to loop through several options at once:

settings = {
    'max_columns': None,
    'min_rows': None,
    'max_rows': 50,
    'precision': 4,
    'float_format': lambda x: f'{x:.2f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)


def pandas_series():
    """
    Introduction to pandas series

    We want to convert the original list into a series

    Key takeaways: 1. The Pandas Series object is something like a powerful version of a Python list,
    or an enhanced version of NumPy array 2. Remember to always contain data consistency.
    :return: a series
    """
    products = ['A', 'B', 'C', 'D']
    products_categories = pd.Series(products)

    daily_rates_dollars = pd.Series([1.07, 1.08, 1.09])
    print(daily_rates_dollars)

    return products, products_categories


def numpy_arrays():
    """

    :return:
    """
    array_a = np.array([1.07, 1.08, 1.09])
    print(array_a)
    print(type(array_a))

    series_a = pd.Series(array_a)
    print(series_a)


def exercises_1():
    """

    :return:
    """
    employee_names = ['Amy White', 'Jack Stewart', 'Richard Lauderdale', 'Sara Johnson']
    print(employee_names)

    employee_names_Series = pd.Series(employee_names)
    print(employee_names_Series)
    print(type(employee_names_Series))

    work_experience_yrs = pd.Series([5, 3, 8, 10])
    print(work_experience_yrs)

    array_age = np.array([50, 53, 35, 43])
    print(array_age)
    series_age = pd.Series(array_age)
    print(series_age)

    # Methods: numerical and categorical:

    employees_work_exp = pd.Series({
        'Amy White': 3,
        'Jack Stewart': 5,
        'Richard Lauderdale': 4.5,
        'Sara Johnson': 22,
        'Patrick Adams': 28,
        'Jessica Baker': 14,
        'Peter Hunt': 4,
        'Daniel Lloyd': 6,
        'John Owen': 1.5,
        'Jennifer Phillips': 10,
        'Courtney Rogers': 4.5,
        'Anne Robinson': 2,
    })

    print(employees_work_exp.head(10))
    print(employees_work_exp.tail())

    data = pd.read_csv('data_regressions/Location.csv')
    location_data = data.copy()
    location_data_series = pd.Series(data=location_data.any())
    print(location_data_series)

    # methods: .len(), .unique(), .nunique(), .describe()
    print(location_data_series.nunique())

    numbers = pd.Series([15, 1000, 23, 45, 444])
    print(numbers.sort_values(ascending=False))

    print(location_data.sort_values(by='Location'))

    # Funktionen in Funktionen ausführen:
    def plus_five(x):
        return x + 5

    def m_by_3(m):
        return plus_five(m) * 3

    print(m_by_3(5))


def exercises_2():
    """

    :return:
    """

    def distance_from_zero(x):
        try:
            result = abs(x)
            print(result)
        except BaseException as e:
            print('Not possible - because of:', e)

    distance_from_zero('cat')

    def distance_from_zero_2(x):
        if type(x) is int or type(x) is float:
            result = abs(x)
            print(result)

        else:
            print('Not possible')

    distance_from_zero_2('cat')

    Numbers = [15, 40, 50, 100, 115, 140]
    print(Numbers.index(15))

    Numbers = [15, 40, 50, 100, 115, 140]
    Numbers.sort(reverse=True)
    print(Numbers)

    # split-method for tuples:
    name, age = 'Peter,24'.split(',')

    print(name)
    print(age)

    def rectangle_info(a, b):
        area = a * b
        perimeter = 2 * (a + b)
        return area, perimeter

    area, perimeter = rectangle_info(2, 10)
    print(area)
    print(perimeter)


    # while-loops:
    num = 0
    while num <= 29:
        num += 1
        # print(num)
        if num % 2 != 0:
            print(num, end=' ')
        else:
            continue

    # for loops and range()
    x = [x for x in range(1, 11)]
    print(x)

    for item in x:
        print(item * 2, end=' ')

    num = list(range(1, 31))

    for i in num:
        if i % 2 != 0:
            print(i, end=' ')
        else:
            print('Even', end=' ')

    # komplett sophisticated loop:
    # IMPORTANT is the indexing in the last line - otherwise we would start mit ZERO
    n = [1, 2, 3, 4, 5, 6]

    for item in range(len(n)):
        print(n[item] * 10)

    prices = {
        "box_of_spaghetti": 4,
        "lasagna": 5,
        "hamburger": 2
    }
    quantity = {
        "box_of_spaghetti": 6,
        "lasagna": 10,
        "hamburger": 0
    }

    money_spent = 0

    for i in prices:
        if prices[i] >= 5:
            money_spent += prices[i] * quantity[i]
    print(money_spent)

    prices = {
        "box_of_spaghetti": 4,
        "lasagna": 5,
        "hamburger": 2
    }
    quantity = {
        "box_of_spaghetti": 6,
        "lasagna": 10,
        "hamburger": 0
    }

    money_spent = 0

    prices = {
        "box_of_spaghetti": 4,
        "lasagna": 5,
        "hamburger": 2
    }
    quantity = {
        "box_of_spaghetti": 6,
        "lasagna": 10,
        "hamburger": 0
    }

    money_spent = 0

    for i in prices:
        if prices[i] < 5:
            money_spent += prices[i] * quantity[i]
    print(money_spent)


    # für while-loops musst du immer zuerst einen Index bestimmen:

    nums = [1,35,12,24,31,51,70,100]

    index = 0
    count = 0

    while index < len(nums):
        if nums[index] < 20:
            count += 1
        index += 1
    print(count)


def pandas_dataframes():
    """
    A series is a single column of data and corresponds to a single variable
    A dataframe contains multi-column data - every column contains data of its own type - it's a
    collection of multiple series

    You can regard it analytically as a collection of multiple observations for the given variables.

    We need the row- and the column-index for referencing a certain point

    Df is like an enhanced python dictionary: we can provide a whole object that contains the values of an entire
    column to the dictionary keys. Inherits the characteristics of a python dictionary.
    :return:
    """

    # Variant 1 - create a dictionary called data where the values should be lists. :


    data = {
        'Name': ['Amy White', 'Jack Stewart', 'Richard Lauderdale', 'Sara Johnson'],
        'Age': [50, 53, 35, 43],
        'Working Experience (Yrs.)': [5, 8, 3, 10]
    }

    df = pd.DataFrame(data=data, index=[0, 1, 2, 3])

    print(df)

    # Variant 2 - create a list named data which should only contain dictionaries as its elements:
    #  When we create Dataframe from a list of dictionaries, matching keys will be the columns and corresponding
    #  values will be the rows of the Dataframe. If there are no matching values and columns in the dictionary,
    #  then the NaN value will be inserted into the resulting Dataframe.

    data2 = [
        {'Name': 'Amy White', 'Age': 50, 'Working Experience (Yrs.)': 5},
        {'Name': 'Jack Stewart', 'Age': 53, 'Working Experience (Yrs.)': 8},
        {'Name': 'Richard Lauderdale', 'Age': 35, 'Working Experience (Yrs.)': 3},
        {'Name': 'Sara Johnson', 'Age': 43, 'Working Experience (Yrs.)': 10}

    ]

    df2 = pd.DataFrame(data=data2, index=[0, 1, 2, 3])
    print(df2)


    # Variant 3 - create a dictionary called data where the values should be pandas Series.
    # Store the pandas Series in variables named names, age, and working_experience_yrs.

    names = pd.Series(['Amy White', 'Jack Stewart', 'Richard Lauderdale', 'Sara Johnson'])
    age = pd.Series([50, 53, 35, 43])
    working_experience_yrs = pd.Series([5, 8, 3, 10])

    data3 = {'Names': names, 'Age': age, 'Working Experience (Yrs.)': working_experience_yrs}

    df3 = pd.DataFrame(data=data3, index=[0, 1, 2, 3])
    print(df3)

    # Variant 4 - create a list called data where the values should be lists.
    ls1 = ['Amy White', 50, 5]
    ls2 = ['Jack Stewart', 53, 8]
    ls3 = ['Richard Lauderdale', 35, 3]
    ls4 = ['Sara Johnson', 43, 10]

    data4 = [ls1, ls2, ls3, ls4]

    df4 = pd.DataFrame(data=data4, columns=['Name', 'Age', 'Working Experience (Yrs.)'])
    print(df4)

    # A revision to pandas DataFrames:

    array_a = np.array([[3, 2, 1], [6, 3, 2]])
    #print(array_a)

    df_array_a = pd.DataFrame(array_a)
    print('DF wird angelegt - man sieht fortlaufende Indexnummern und Spaltennummern: ')
    #print(df_array_a)

    df_array_a_better = pd.DataFrame(data=array_a, columns=['Column 1', 'Column 2', 'Column 3'],
                                     index=['Row 1', 'Row 2'])
    print('Verbesserter DF mit expliziten Spalten und Reihen(Index)-Namen:')
    #print(df_array_a_better)

    df_lending = pd.read_csv('data_regressions/Lending-company.csv', index_col='LoanID')
    lending_co_data = df_lending.copy()
    print(lending_co_data)

    """
    COMMON ATTRIBUTES:
    """
    print(lending_co_data.index) ### Index sind rows
    print(lending_co_data.columns)
    print(lending_co_data.axes) # damit sieht man Spalten und Reihen :-)

    # G E N I A L !
    print(lending_co_data.dtypes)

    # values vs. to_numpy:
    print(lending_co_data.values)
    print(lending_co_data.to_numpy())

    print(lending_co_data.shape)


def dataframe_indexing():
    """
    Data selection or subset-selection
    :return:
    """

    data = pd.read_csv('data_regressions/Lending-company.csv', index_col='StringID')
    lending_co_data2 = data.copy()
    print(lending_co_data2.head())

    # clever selecting by just the column-label:
    print(lending_co_data2.Product.head(5))
    print(lending_co_data2.Location.head(5))

    # alternative solution:
    print(lending_co_data2['Product'].head(5))
    print(lending_co_data2['Location'].head(5))

    # Series-object vs.
    print('We see a series-object:', type(lending_co_data2['Location']))
    print(lending_co_data2[['Location']])
    print("""We want to obtain a DataFrame even if it is only one column, we use double brackets:
            a nested list containing a single element.""")
    print(type(lending_co_data2[['Location']]))

    # we extract a dataframe from a dataframe:
    """
    Genaus so ordne ich Spalten neu :-)
    """
    print(lending_co_data2[['TotalPrice', 'Deposit', 'LoanStatus']])

    # A more elegant way to obtain the same result is to store the list containing the column names
    # in a separate variable:
    prod_loc = ['Location', 'Product']
    print(lending_co_data2[prod_loc].head(5))

    try:
        print(lending_co_data2['Location', 'Product'])
    except BaseException as e:
        print(f'You will see a key-error since we forgot to use double-brackets: {e}')

    """
    IMPORTANT: the iloc-Indexer - iloc accessor
    """

    # row specifier - row indexer:
    print(lending_co_data2.iloc[56])
    # better practice: indicate explicit all columns with ":"
    print(lending_co_data2.iloc[56, :])

    print(lending_co_data2.iloc[0:10])

    # column indexer:
    # convention: always use the accessor's specifiers in pairs:
    print(lending_co_data2.iloc[:, 4])

    # value specifier - value indexer - works like a cursor:
    print(lending_co_data2.iloc[78, 1])

    # specifying the second AND fourth row or column
    print(lending_co_data2.iloc[[1, 3], :])
    print(lending_co_data2.iloc[:, [1, 3]])

    """
    The loc[]-accessor:
    referring by its index-labels
    """
    print('We see the whole row:\n', lending_co_data2.loc['LoanID_78'])
    # it shows the same - but it is a more readable line of code:
    print("""
        loc['LoanID_78', :] has been designed to let you take advantage of the explicit index
        and column labels of your data table - with both features it is more readable...
    """)
    print(lending_co_data2.loc['LoanID_78', :])

    print('Now we see the cross-product of Row-label (Index) and Column-label: ')
    print(lending_co_data2.loc['LoanID_3', 'Region'])

    # Hint: you will see the same result:
    print(lending_co_data2['Location'])
    print(lending_co_data2.loc[:, 'Location'])
    

    """    
           Constructing DataFrame from a dictionary.
    >>> d = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = pd.DataFrame(data=d)
    >>> df
       col1  col2
    0     1     3
    1     2     4
    Notice that the inferred dtype is int64.
    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object
    To enforce a single dtype:
    >>> df = pd.DataFrame(data=d, dtype=np.int8)
    >>> df.dtypes
    col1    int8
    col2    int8
    dtype: object
    Constructing DataFrame from a dictionary including Series:
    >>> d = {'col1': [0, 1, 2, 3], 'col2': pd.Series([2, 3], index=[2, 3])}
    >>> pd.DataFrame(data=d, index=[0, 1, 2, 3])
       col1  col2
    0     0   NaN
    1     1   NaN
    2     2   2.0
    3     3   3.0
    """


if __name__ == '__main__':
    #products_ret, products_categories_ret = pandas_series()
    #print('Classic python list:', products_ret)
    #print('Pandas series: \n', products_categories_ret)
    #numpy_arrays()
    #exercises_1()
    #exercises_2()
    pandas_dataframes()
    #dataframe_indexing()

