import pandas as pd
import numpy as np
import seaborn

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import scipy.stats as stat
import statsmodels.api as sm

seaborn.set()
#style.use('ggplot')
#style.use('fivethirtyeight')


class LinearRegression(LinearRegression):
    """
    This amendment overrides the LinearRegression-class and adds a more convenient way to compute p-values
    in sklearn.

    LinearRegression class after sklearn's, but calculate t-statistics and p-values for model coefficients (betas).
    Additional attributes available after .fit()-method are 't' and 'p' which are of the shape
    (y.shape[1], X.shape[1]) which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it in X.
    """
    # nothing changes in __init__
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1, positive=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, x, y, n_jobs=1):
        """

        :param x:
        :param y:
        :param n_jobs:
        :return:
        """
        self = super(LinearRegression, self).fit(x, y, n_jobs)

        # Calculate SSE (sum of squared errors) and SE (standard errors)
        sse = np.sum((self.predict(x) - y) ** 2, axis=0) / float(x.shape[0] - x.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(x.T, x))))])

        # compute the t-statistic for each feature
        self.t = self.coef_ / se

        # find the p-value for each feature
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - x.shape[1])))
        return self


def test_pvalues():
    """
    For testing the overridden class LinearRegression
    :return:
    """
    data = pd.read_csv('data_regressions/1.02.+Multiple+linear+regression.csv')
    print(data.describe())
    x = data[['SAT', 'Rand 1,2,3']]
    y = data['GPA']

    # we create the regression but use the overridden class
    reg_with_pvalues = LinearRegression()
    reg_with_pvalues.fit(x, y)

    # but we can now check the p-values of the x's - what's contained in the local var. 'p' in an instance of
    # LinearRegression()-class
    print(reg_with_pvalues.p)

    # Let's create a new data frame with the names of the features
    reg_summary = pd.DataFrame([['SAT'], ['Rand 1,2,3']], columns=['Features'])
    # We create and fill a second column, called "Coefficients" of the regression
    reg_summary['Coefficients'] = reg_with_pvalues.coef_
    # Finally, we add the p-values we just calculated
    reg_summary['p-values'] = reg_with_pvalues.p.round(3)

    print(reg_summary)


def linear_regression_supervised_learning_with_known_targets():
    """
    Supervised learning, where we got inputs and targets. Our algorithm will find the optimal coefficients
    of a linear regression model
    :return:
    """
    data = pd.read_csv('data_regressions/1.01.+Simple+linear+regression.csv')

    x = data['SAT']  # now called feature
    y = data['GPA']  # now called target

    print('Result: 1D-vectors of a certain length (sklearn will need 2D-arrays):', x.shape, y.shape)

    # we create the object:
    # you can omit the arguments since they are default, for teaching purposes: we make a safety-copy, fit the
    # x0-implicit variable automatically and use just one CPU for the job, since we have just little data
    reg = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
    #reg.fit(x, y)
    #print(reg)

    print('We get a value error and should follow the hint:\nValueError: Expected 2D array, got 1D array instead:\n'
          'Reshape your data either using array.reshape(-1, 1) if your data has a single feature or '
          'array.reshape(1, -1) if it contains a single sample. ')

    x_matrix = x.values.reshape(-1, 1)
    print(x_matrix)
    reg.fit(x_matrix, y)
    print(reg)

    """
    Interpreting the results of the object reg:
    """
    print('R-squared:', reg.score(x_matrix,y))
    print('Coefficients:', reg.coef_)
    print('Intercept:', reg.intercept_)

    # Making predictions:
    # print(reg.predict(x_matrix))
    new_data = pd.DataFrame(data=[1740, 1760], columns=['SAT'])
    print(reg.predict(new_data))

    new_data['Predicted_GPA'] = reg.predict(new_data)
    print(new_data)

    # yhat = 0.2750 + 0.0017 * x1
    plt.scatter(x, y)
    yhat = reg.intercept_ + reg.coef_ * x_matrix
    fig = plt.plot(x, yhat, label="Regression line", lw=4, c='orange')
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('GPA', fontsize=20)
    plt.legend()
    plt.show()


def adjusted_r2(X, y, regression):
    r2 = regression.score(X, y)
    n = X.shape[0]
    p = X.shape[1]
    rsq_adj = 1 - (1-r2) * (n - 1) / (n - p - 1)
    return rsq_adj


def multiple_linear_regression():
    """
    Create a multiple linear regression and discuss the outcome.
    Display the intercept and coefficients
    Find the R-squared and Adjusted R-squared - compare and also compare with the R-squared of the simple
    linear regression
    Predict prices for different appartment sizes
    Find the multivariate p-values with the F-Regression and your ammended class - discuss
    :return:
    """
    data = pd.read_csv('data_regressions/real_estate_price_size_year.csv')
    x = data[['size', 'year']]
    y = data['price']
    print(x.shape)

    # Regression
    reg = LinearRegression()
    reg.fit(x, y)
    print(f'Intercept: {reg.intercept_}, Coeffs: {reg.coef_}, R2: {reg.score(x, y)}')

    adjr2 = adjusted_r2(x, y, reg)
    print(adjr2)

    size_to_predict = 750
    year_to_predict = 2009
    print(f'The price of a house from {year_to_predict} with a size of {size_to_predict} '
          f'sqrft is {reg.predict([[size_to_predict, year_to_predict]])}')

    """
    Feature selection (F-regression): How to detect the variables which are unneeded in a model?
    Feature selection simplifies models.
    We can disregard every feature (variable) with a p-value > 0.05
    
    As a workaround we need the module and function feature_selection.f_regression: F-regression creates simple linear
    regressions of each feature and the dependent variable
    """

    # the output shows two arrays: first, F-statistics, second, p-values
    print(f_regression(x, y))
    p_values = f_regression(x, y)[1]
    print(f'The p-values for X are: {p_values.round(3)}')

    reg_summary = pd.DataFrame([['Size'], ['Year']], columns=['Features'])
    reg_summary['Coefficients'] = reg.coef_
    reg_summary['P-values'] = p_values.round(3)
    print(reg_summary)

    """
    As suggested in the previous lecture, the F-regression does not take into account the interrelation of the
    features.
    A not so simple fix for that is to amend the LinearRegression() class.
    Use the above amended class
    
    Note that the results will be identical to those of StatsModels.
    """

    reg_with_pvalues = LinearRegression()
    reg_with_pvalues.fit(x, y)

    # but we can now check the p-values of the x's - what's contained in the local var. 'p' in an instance of
    # LinearRegression()-class
    print(reg_with_pvalues.p)

    # Let's create a new data frame with the names of the features
    reg_summary_from_class = pd.DataFrame([['Size'], ['Year']], columns=['Features'])
    # We create and fill a second column, called "Coefficients" of the regression
    reg_summary_from_class['Coefficients'] = reg_with_pvalues.coef_
    # Finally, we add the p-values we just calculated
    reg_summary_from_class['p-values'] = reg_with_pvalues.p.round(3)

    print(reg_summary_from_class)

    print("""
        Using F-regressions, it seems that 'Year' is not even significant, therefore we should remove it
        from the model.
        But using our ammended class the p-value of 'Year' seems very significant.
    """)

    """
    Feature scaling: - Standardization of the features (weights)
    
    In ML we call coefficents weights - the more important they are the more weight they add to the model
    The bigger the weight, the bigger the impact!
    The ML word for intercept is "bias": the idea is if we need to adjust our regression by some number/constant/
    intercept, then the model is biased :-)
    
    The process of transforming data into a standard scale 
    variabel - mean of original var divided by stdev of orig. var.
    So we force the features to appear similar
    Import and use a standard scaler module from sklearn
    
    Feature selection by p-values or F-regression can now be omitted :-) since a variable with a very small weight
    is penalized by itself - that's why you won't compute p-values in ML-practice in case of you standardized your
    data.
    """

    scaler = StandardScaler()
    # fit your input into the instance:
    scaler.fit(x)
    # normalize the fitted features:
    x_scaled = scaler.transform(x)
    # print(x_scaled)
    reg_scaled = LinearRegression()
    reg_scaled.fit(x_scaled, y)
    print(f'Intercept: {reg_scaled.intercept_}, Coeffs: {reg_scaled.coef_}, R2: {reg_scaled.score(x, y)}')
    reg_scaled_summary = pd.DataFrame([['Intercept aka Bias'], ['Size'], ['Year']], columns=['Features'])
    reg_scaled_summary['Weights'] = reg_scaled.intercept_, reg_scaled.coef_[0], reg_scaled.coef_[1]
    print(reg_scaled_summary)

    # Make predictions with the standardized coefficients (weights)
    data_to_predict = pd.DataFrame([[750, 2009]], columns=['size', 'year'])
    data_to_predict_scaled = scaler.transform(data_to_predict)

    print('You see exactly the same result with normalized data like above calculated with plain data:')
    print(reg_scaled.predict(data_to_predict_scaled))


def exercise_train_test_split():
    """
    Train - Test - Split

    We suggest to split our data into training and testing parts: We train the model on the training dataset
    but then test it on the testing dataset

    from sklearn.model_selection import train_test_split
    :return:
    """

    # Generate some data we are going to split
    # Numpy arange is equivalent to python's "range()" but displays the values in an array:
    a = np.arange(1, 101)

    # another ndarray - the numbers are intentionally picked for easy comparing - obviously, the difference
    # between the elements of the two arrays is 500 for any two corresponding elements
    b = np.arange(501, 601)
    print(a)
    print(b)

    # Split the data:
    # train_test_split(x) splits arrays or matrices into random train and test subsets
    print(train_test_split(a))

    # we need the two arrays in two variables:
    a_train, a_test = train_test_split(a)
    print(a_train)
    print(a_test)
    print(f'Shape of a_train {a_train.shape}, shape of a_test {a_test.shape} - default split are 72/25 or 80/20')

    # normally we want to shuffle our data BUT in terms of time series (stock prices) we must not shuffle:
    a_train, a_test = train_test_split(a, test_size=0.2, shuffle=False)
    print(a_train)
    print(a_test)

    # Most often, we have inputs and targets, so we have to split 2 different arrays. We are simulating
    # this situation by splitting 'a' and 'b'
    # You can specify the 'test_size' with an argument
    # Finally, you should always employ a 'random_state'. In this way you ensure that when you are splitting
    # the data you will always get the SAME random shuffle

    # Note 2 arrays will be split into 4, the order is train1, test1, train2, test2
    # It is very useful to store them in 4 variables, so we can later use them
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

    print("""
        obviously, the difference
        between the elements of the two arrays is 500 for any two corresponding elements
    """)
    print(a_train)
    print(a_test)
    print(b_train)
    print(b_test)


if __name__ == '__main__':
    linear_regression_supervised_learning_with_known_targets()
    test_pvalues()
    multiple_linear_regression()
    exercise_train_test_split()


