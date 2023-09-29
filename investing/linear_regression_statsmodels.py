import pandas as pd
import numpy as np
import seaborn
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

seaborn.set()
#style.use('ggplot')
#style.use('fivethirtyeight')


def single_linear_regression():
    """

    :return:
    """
    data = pd.read_csv('data_regressions/1.01.+Simple+linear+regression.csv')

    print(data)
    print(data.describe())

    y = data['GPA']
    x1 = data['SAT']

    """
    plt.scatter(x1, y)
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('GPA', fontsize=20)
    plt.show()
    """

    # Building the regression model:
    # add a constant called x which is always 1 and gives the intercept a storage box
    # our regression is supposed to estimate the intercept-constant
    # x ist eine Hilfsvariable und in der Regressionsgleichung können wir sie als x0 denken (implizite Variable)

    x = sm.add_constant(x1)
    results = sm.OLS(y, x).fit()
    print(results.summary())

    yhat = 0.2750 + 0.0017 * x1
    plt.scatter(x1, y)
    fig = plt.plot(x1, yhat, label="Regression line", lw=4, c='orange')
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('GPA', fontsize=20)
    plt.legend()
    plt.show()


def multi_linear_regression_nonsense_variable():
    """
    This function shows how an additional variable increases the R-squared value (Ratio of explained var. and total
    variability) but the Adjusted R-squared decreases since our second variable is nonsense and has no predictive
    power. So be careful with excessive use of variables just for the sake of increasing the complexity of your model.
    Additional variables always have to be meaningful.

    Be aware: we can't plot a multi linear regression in a 2D-room! Beyond 3D there is no way of visual representation.
    :return:
    R-squared: Quotient/ratio of SSR (variability explained by regression) divided by sum of squares total
    Adjusted R-squared: penalizes excessive use of variables, smaller than R^2
    F-statistic: Compare the F-numbers of different models - the higher number is the more significant model. The lower
    the F-Statistic the closer to a non-significant model.

    OLS-Assumptions (know all of them and take into consideration before modelling regressions:

    1. Linearity
    2. No endogeneity
    3. Normality and homoscedasticity
    - if the x-values scatter too much from left to right then compute the log, like np.log(df['x']) - semi-log-model
        - yhat = b0 + b1(logx1)
        - log-yhat = b0 + b1x1
    - if the y-values scatter too much from left to right then compute the log, like np.log(df['y']) - log-log-model
        - log-yhat = b0 + b1(logx1)
        - as X increases by 1 percent, Y increases by b1 percent :-)
    4. No autocorrelation: No identifiable relationship should exist between the values of the error term.
    5. No multicollinearity: No predictor variable should be perfectly or almost perfectly explained by the other
    predictors
    """
    df = pd.read_csv('data_regressions/1.02.+Multiple+linear+regression.csv')

    # create multiple regression
    y = df['GPA']
    x1_nonsense = df[['SAT', 'Rand 1,2,3']]
    x1 = df['SAT']

    x = sm.add_constant(x1_nonsense)
    results = sm.OLS(y, x).fit()
    results_table = results.summary()

    print('\nCompare the regressions of x1 and x1_nonsense. You clearly see how the adjusted R-squared decreases \n'
          'in the latter one due to the fact that the extra variable put no additonal meaning and predictive\n'
          'power into the model. Also look at the p-value (smallest level of significance) It is extremely high \n'
          'so we cannot reject the H0 at 76%.')

    return results_table


def dummy_variables():
    """
    Dummy variables or how to deal with categorical predictors
    :return:
    """
    raw_data = pd.read_csv('data_regressions/Dummies.csv')

    # map the data with numerical substitutes/dummies
    # we use copy() instead of inplace-argument
    data = raw_data.copy()
    data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0})

    # Regression
    # Following the regression equation, our dependent var. (y) is the GPA
    y = data['GPA']

    # Similarly, our independent var. (x) is the SAT score
    x1 = data[['SAT', 'Attendance']]

    # Add a constant. Essentially, we are adding a new column (equal in length to x), which consists only of 1s
    x = sm.add_constant(x1)

    # Fit the model, according to the OLS method with a dependent var. y and an independent
    results = sm.OLS(y, x).fit()
    print(results.summary())

    # Create a scatter plot of SAT and GPA
    # for the point with Attendance we use a different color from a color-map:
    # Create one scatter plot with all observations, use the series Attendance as color, and choose a cmap
    # of your choice (totally arbitrary)
    plt.scatter(data['SAT'], y, c=data['Attendance'], cmap='RdYlGn_r')

    # Define the two regression equations, depending on whether they attended (yes), or didn't (no)
    yhat_no = 0.6439 + 0.0014 * data['SAT']
    yhat_yes = 0.6439 + 0.0014 * data['SAT'] + 0.2226 * 1 # die 1 könnten wir auch weglassen und die beiden const addie.

    # Original regression line - we don't see the difference made by Attendance - we omit this variable
    yhat = 0.2750 + 0.0017 * data['SAT']

    # Plot the two regression lines - solo coz we wanna see the difference attendance makes
    fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837', label='Not attended')
    fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='#a50026', label='Attended')
    # Plot the original regression line
    fig = plt.plot(data['SAT'], yhat, lw=3, c='#4C72B0', label='Original, only Var. "SAT"')

    # Name your axes :)
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('GPA', fontsize=20)
    plt.legend()
    plt.show()

    """
    How to make predictions based on the regressions we create - compare doing it by the plain regression equations
    and a statsmodels-method
    """

    # Let's see what's inside the independent variable.
    # The first column comes from the "add_constant"-method. It's only 1s
    print(x)

    # Create a new data frame, identical in organization to X.
    # The constant is always 1, while each of the lines corresponds to an observation (student)
    new_data = pd.DataFrame({'const':1, 'SAT': [1700, 1670], 'Attendance': [0, 1]})

    # By default, in older Pandas-versions the columns are ordered alphabetically. But if you feed the model wrong
    # you will get wrong results. So, to be on the save side, reorder them intentionally
    new_data = new_data[['const', 'SAT', 'Attendance']]
    print(new_data)

    # Renaming the indices just for teaching purposes. That's not a good practice! If we want to use NumPy, sklearn etc.
    # methods on a df with renamed indices, they will simply be lost and returned to 0, 1, 2, 3 etc.
    new_data.rename(index={0: 'Bob', 1: 'Alice'}, inplace=True)
    print(new_data)

    # Use the predict-method on the regression with the new data as a single argument
    predictions = results.predict(new_data)
    print('\nYou see clearly that Alice obtained a higher grade due to her attendance despite she had a lower SAT:')
    print(predictions)
    print('\nNow we cross check it by plugging in the data in the plain equation yhat = intercept + b1x1 + b2x2')
    yhat_Bob = 0.6439 + 0.0014 * new_data.loc['Bob', 'SAT']
    print('GPA Bob: ', yhat_Bob)
    yhat_Alice = 0.6439 + 0.0014 * new_data.loc['Alice', 'SAT'] + 0.2226
    print('GPA Alice: ', yhat_Alice)


def multiple_regression_exercise():
    """
    It's not possible to plot in 2D the results of this exercise since the sizes of the variables are too different
    It is possible to plot the regression line which is fed by all predictor-variables but only y and one var as point.
    The result is wrong... debug it.
    :return:
    """
    raw_data = pd.read_csv('data_regressions/real_estate_price_size_year_view.csv')
    data = raw_data.copy()
    data['view'] = data['view'].map({'No sea view': 0, 'Sea view': 1})
    # print(data)
    y = data['price']
    x1 = data[['size', 'year', 'view']]
    x = sm.add_constant(x1)
    model = sm.OLS(y, x).fit()
    print(model.summary())

    yhat = -5398000 + 56730 + data['size'] * 223.0316 + data['year'] * 2718.9489
    plt.scatter(data[['size']], y)
    plt.plot(data['size'], yhat, lw=2, c='blue')
    plt.show()


def multiple_regression_exercise_size_year():
    """
    It's not possible to plot in 2D the results of this exercise since the sizes of the variables are too different
    It is possible to plot the regression line which is fed by all predictor-variables but only y and one var as point.
    The result is wrong... debug it.
    :return:
    """
    raw_data = pd.read_csv('data_regressions/real_estate_price_size_year.csv')
    data = raw_data.copy()
    # print(data)
    y = data['price']
    x1 = data[['size', 'year']]
    x = sm.add_constant(x1)
    model = sm.OLS(y, x).fit()
    print(model.summary())

    yhat = -5398000 + 56730 + data['size'] * 223.0316 + data['year'] * 2718.9489
    plt.scatter(data[['size']], y)
    plt.plot(data['size'], yhat, lw=2, c='blue')
    plt.show()


if __name__ == '__main__':
    single_linear_regression()
    result = multi_linear_regression_nonsense_variable()
    print(result)
    dummy_variables()
    multiple_regression_exercise()
    multiple_regression_exercise_size_year()