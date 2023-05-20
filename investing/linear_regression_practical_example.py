import os

import numpy as np
import pandas as pd

import scipy.stats as stat
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

sns.set()

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
    'min_rows': 40,
    'max_rows': 50,
    'precision': 2,
    'float_format': lambda x: f'{x:.3f}'
    }

for option, value in settings.items():
    pd.set_option("display.{}".format(option), value)


def linear_regression_practical_example():
    """
    Goal: We would like to predict the price of a used car depending on its specifications

    - Find the potential regressors (features, input): e. g. brand, mileage, engine-volume, year of production
    1.Step: Preprocessing - data cleaning and cleansing
    :return:
    """

    if not os.path.exists('statics/pictures'):
        os.makedirs('statics/pictures')

    raw_data = pd.read_csv('data_regressions/1.04.+Real-life+example.csv')
    print(raw_data)

    # with argument include:'all' we also see categorical data - default is only numerical data
    # registration e.g. has nearly 4k top-entries 'Yes' - so the distribution and thus variable is useless for us
    print(raw_data.describe(include='all'))

    # 1. Preprocessing - Data cleaning and cleansing:
    # we won't need the Model - too many distinct values, too many dummies:
    # determining the vars. of interest
    data_1 = raw_data.drop(['Model'], axis=1)
    print(data_1)

    # Dealing with missing values:
    # price and engine volume have missing values - you see it in describe()-count

    # False means is filled, True means isnull
    # you see 172 missing prices and 150 missing EngineV - True is 1 hence we can use sum() :-)
    print(data_1.isnull().sum())
    # delete - it's okay to delete up to 5% of the observations
    data_no_mv = data_1.dropna(axis=0)
    print(data_no_mv.isnull().sum())

    # you see: count of every column is equal now, so no more missing values
    print(data_no_mv.describe(include='all'))

    # the max-values seem to be partly way off in respect to the mean and quartiles:
    # print the distributions (PDFs) to visualize it:

    # Dealing with outliers:

    # we should expect a normal dist. but will encounter kind of an exponential one - we have to
    # get rid of the outliers:
    # sns.displot(data_no_mv['Price'], kde=True)
    #plt.show()

    # we do it by the quantile-method with quantile as only argument
    q = data_no_mv['Price'].quantile(0.99)
    print('Value of the upper limit of the 0.99 % quantile:', q.round(2))
    data_2 = data_no_mv[data_no_mv['Price'] < q]
    print(data_2)
    print(data_2.describe(include='all'))

    q = data_2['Mileage'].quantile(0.99)
    data_3 = data_2[data_2['Mileage'] < q]
    print(data_3.describe(include='all'))

    sns.displot(data_3['Mileage'], kde=True)
    plt.savefig("statics/pictures/distribution_mileage.png", facecolor='white', edgecolor="white", dpi=300)
    #plt.show()

    sns.displot(data_2['Price'], kde=True)
    plt.savefig("statics/pictures/distribution_price.png", facecolor= 'white', edgecolor="white", dpi=300)

    # examine manually the strange values of Engine-Volume:
    engv = pd.DataFrame(raw_data['EngineV'])
    engv = engv.dropna(axis=0)
    print(engv.sort_values(by='EngineV').tail(50))
    data_4 = data_3[data_3['EngineV'] < 6.5]
    # sns.displot(data_4['EngineV'], kde=True)

    # Subplots with Seaborn:
    # define plotting region (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2)

    # create distribution-plot in each subplot - if we work with subplots and axes, we need to use histplot()
    sns.histplot(data_no_mv['Price'], kde=True, ax=axes[0, 0])
    sns.histplot(data_3['Mileage'], kde=True, ax=axes[0, 1])
    sns.histplot(data_2['Price'], kde=True, ax=axes[1, 0])
    sns.histplot(data_4['EngineV'], kde=True, ax=axes[1, 1])

    plt.show()

    # Year: the problem is at the lower end - vintage cars are the outliers - so we take the 1-Percentile
    # and keep all rows above 1%
    q = data_4['Year'].quantile(0.01)
    data_5 = data_4[data_4['Year'] > q]

    print(data_5.describe(include='all'))

    data_cleaned = data_5.reset_index(drop=True)
    print(data_cleaned.describe(include='all'))

    # Check the OLS assumptions:
    # 1. Check for linearity:
    # let's plot the datapoints of feature-targets-pairs for that we can see the distributions
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
    ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
    ax1.set_title('Price and Year')
    ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
    ax2.set_title('Price and EngineV')
    ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
    ax3.set_title('Price and Mileage')
    plt.savefig("statics/pictures/scatterplots_target_features.png", facecolor='white', edgecolor="white", dpi=300)
    # plt.show()

    # we can spot exponential curves with the year, enginev and mileage and have to transform the variables
    log_price = np.log(data_cleaned['Price'])
    data_cleaned['log_price'] = log_price
    # print(data_cleaned[['Price', 'log_price']])

    # 1. Check for linearity - now with log-price-transformation:
    # let's plot the datapoints of feature-targets-pairs for that we can see the distributions
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
    ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
    ax1.set_title('Log-Price and Year')
    ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
    ax2.set_title('Log-Price and EngineV')
    ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
    ax3.set_title('Log-Price and Mileage')
    plt.savefig("statics/pictures/scatterplots_target_features_log_transformed.png", facecolor='white', edgecolor="white", dpi=300)
    plt.show()

    data_cleaned.drop(['Price'], axis=1, inplace=True)
    print(data_cleaned)

    # 2. Check for no endogeneity:
    # omitted variable bias: we must not forget a crucial variable otherwise the error-term would enlarge
    # in our example the brand Porsche could explain high prices of small cars with a high mileage
    # so we must not forget the brand as a feature
    # Todo: for homework in a second step when we hone the original model

    # 3. Check for normality, zero mean and homoscedasticity:
    # homoscedast. holds true since we already log-transformed

    # 4. Check for no autocorrelation - True, we have no time series, just a snapshot of the moment

    # 5. Check for no multicollinearity
    # we check it with the help of statsmodels vif-function (Variance Inflation Factor):

    print(data_cleaned.columns.values)
    variables = data_cleaned[['Mileage', 'EngineV', 'Year']]
    vif = pd.DataFrame()
    # List comprehension :-)
    # let's check for the intercorrelation of the variables
    vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif['Features'] = variables.columns
    print("""
        The VIF may be calculated for each predictor by doing a linear regression of that predictor on 
        all the other predictors, and then obtaining the R2 from that regression. The VIF is just 1/(1-R2).
        Normally a very high R2 is good, like 80 or 90% - but vice versa this implies that two variables
        with such a high number are strongly correlated - an R2 of 0.8 means a VIF of 5 (1/0.2)
        
        VIF element of 1 until infinity
        If VIF is 1 (minimum) there is no multicollinearity at all, values between 1 and 5 are perfectly fine,
        Some sources state that a VIF above 5 is unacceptable, others put the boundary at six, some say
        10 is a cutoff line. So there is no common ground and opinion - no firm consensus on the topic.
        In our example it seems that year is TOO correlated with the other variables - so we can omit it
        
        A Note on Multicollinearity

        What most people are wondering is not 'when do we have multicollinearity' as we usually do have some.
        
        The better question to ask is 'When can we safely ignore multicollinearity'.
        
        Here's a great article on the topic: http://statisticalhorizons.com/multicollinearity

        ***

        Regarding the variance inflation factor method that we employed from StatsModels, 
        you can read the full documentation here: 
        https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html 
    """)
    print(vif)
    data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
    print(data_no_multicollinearity)

    # Creation of dummy variables:
    """
    Convert categorical variable into dummy/indicator variables.
    Each variable is converted in as many 0/1 variables as there are different values. Columns in the output are each 
    named after a value; if the input is a DataFrame, the name of the original variable is prepended to the value.
    :param drop_first=True - we skip the first categorical var. for avoiding multicollinearity
    """
    data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
    print(data_with_dummies)

    # Rearrange a bit the dataframe
    print(data_with_dummies.columns.values)

    cols = ['log_price', 'Mileage', 'EngineV',  'Brand_BMW', 'Brand_Mercedes-Benz',
             'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
             'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
             'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
             'Registration_yes']

    data_preprocessed = data_with_dummies[cols]
    print(data_preprocessed)

    """
    Linear regression model
    """
    # Declare the variables
    features = data_preprocessed.drop(['log_price'], axis=1)
    targets = data_preprocessed['log_price']

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(features)
    # it is not usually recommended to standardize all feature-vars. - later we learn how to implement a custom scaler
    features_scaled = scaler.transform(features)

    # Train-Test-split
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, targets, test_size=0.2, random_state=42)

    # Create the regression
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    print(f'Bias: {reg.intercept_}, Weights: {reg.coef_}, R2: {reg.score(x_train, y_train)}')

    # A simple way to check the final result is to plot the predicted values against the observed values
    # we see linearity - it's not perfect (look at the variability) but we can improve our model later
    y_hat = reg.predict(x_train)
    plt.scatter(y_train, y_hat)
    plt.xlabel('Targets (y_train)', size=18)
    plt.ylabel('Predictions (y_hat)', size=18)
    plt.xlim(6, 13)
    plt.ylim(6, 13)

    # Plot the residuals (the differences between targets and predictions, so we use subtraction):
    sns.displot(y_train - y_hat, kde=True, height=7, aspect=1.5)    # height=7, width=1.5 times larger than height)
    plt.title('Residuals PDF: normal, zero mean, but a long left tail', size=18)
    plt.show()
    """
    Meaning: there are certain observations for which (y_train - y_hat) is much lower than the mean,
    a much higher price is predicted than is observed! Amend it later by honing the weights of the dummy-variables
    """
    print(f"""
        Our model is explaining {reg.score(x_train, y_train) * 100} % of the variability of the data
    """)
    reg_summary = pd.DataFrame(features.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    print(reg_summary)
    print("""
        This model is far from interpretable since all features (and predictions) are standardized including the dummies
        Positive weights: they increase the predicted price by multiplication, e. g. the bigger the Engine-volume
        the higher the price
        Negative weights: they decrease the predicted price, e. g. the higher the mileage the lower the price
        
        Situation for dummies: since we dropped one category for each dicrete variabel (e.g. Brand), when all
        included dummies are zero, then the drop dummy is one. 
        
        A positive weight shows that the respective category (Brand) is more expensive than the benchmark Audi
        Homework: find the real benchmark so that all other brand-dummies are negative. This leads to that
        the RealBrandBenchmark-coeff. predicts the highest prices.
        Hint: look at BMW :-)
        
        Note: dummy-weights are ONLY compared to their benchmark and NEVER to the continuous variables.
        
        
    """)
    print(data_cleaned['Brand'].unique())
    # we realize that we dropped "Audi": in features there is no Audi. When all other dummies are zero,
    # Audi is 1/True - so Audi is the benchmark

    # which ones are the other benchmarks? We need to look for the missing values in our reg_summary - weights:
    print(data_cleaned['Body'].unique())    # crossover
    print(data_cleaned['Engine Type'].unique())     # Diesel
    print(data_cleaned['Registration'].unique())    # no

    """
    Testing:
    """
    # we start the testing part by finding the predictions:
    y_hat_test = reg.predict(x_test)

    plt.scatter(y_test, y_hat_test, alpha=0.2)
    plt.xlabel('Targets (y_test)', size=18)
    plt.ylabel('Predictions (y_hat_test)', size=18)
    plt.xlim(6, 13)
    plt.ylim(6, 13)
    plt.show()
    print("""
        Higher prices are better predicted than lower prices: in the scatter plot we see more 
        dispersion at the low end. Alpha-param let us see where we have concentration. Lower prices
        are much more scattered pointing at the fact that we did not predict them very well.""")

    # let's manually explore how our model performed in a dataframe-performance df_pf:

    # we see the log-prices and have to reconvert with the exponentials:
    r2_train = reg.score(x_train, y_train)
    r2_test = reg.score(x_test, y_test)
    print(f"""
        R^2 - trainingsdata: {r2_train}
        R^2 - testdata: {r2_test}
    """)

    df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
    # Compare the predictions with our targets:
    # when we added y_test to df_pf Pandas tried to match the indices
    # we see a lot of NaN since we lost the matching of the incices! So we have to reset_index
    # print(y_test)
    y_test = y_test.reset_index(drop=True)
    y_test = np.exp(y_test)     # with exponential getting rid of log-prices
    df_pf['Target'] = y_test
    df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
    df_pf['Difference %'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100)
    print(df_pf.sort_values(by=['Difference %']).head(30))

    print(df_pf.describe(include='all'))


if __name__ == '__main__':
    linear_regression_practical_example()