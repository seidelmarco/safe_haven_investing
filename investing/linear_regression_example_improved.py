import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression

import scipy.stats as stat

import openpyxl
from openpyxl import writer

import seaborn as sns

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
    'min_rows': None,
    'max_rows': 100,
    'precision': 2,
    'float_format': lambda x: f'{x:.3f}'
    }

for option, value in settings.items():
    pd.set_option("display.{}".format(option), value)


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


def linear_regression_practical_example_improved(quantile_percent_upper: float = 0.99,
                                                 quantile_percent_lower: float = 0.05, price_attr: str = 'log_price',
                                                 year_attr: str = 'on'):
    """
    Goal is to improve the model "Prices of used cars". Prior R2 was around .75. We want to reach a higher number.
    Edit: May 31st R2 = .88 training, .8681 test - Adj R2 training: .8787, .8645 test
    We would like to explain around 90% of the variability of the model.
    In the first model we dropped the car "Model" - maybe we should use it.
    More over we should rethink how many outliers we drop or not - which quantiles will we use?
    Do we need other independent variables (more or less dummies)?
    Which factor did we overlook? Are there damaged cars in the sample data? (Some cars are way too cheap - it's
    depicted in the long left tail of outliers in the prices-distribution plot.
    More over our benchmark-brand should be the most expensive one, and so we should sort it into first place for the
    drop=first parameter.
    Look at the cars with 0.00 mileage: what's wrong with them?
    How can we use the infos from the "Model"?

    How to improve the model: 1. Use a different set of vars. 2. Remove a bigger part of outliers. 3. Use different
    kinds of transformations.

    1. Delete all Price-Na's
    2. Plot distributions of every brand and find outliers - delete outliers
    3. Find the benchmark of the unique categories, reorder and drop the first dummy (benchmark-dummy)
    4. Choose the independent vars.
    5. Check for multicollinearity: is there a biasing correlation between Mercedes, mileage and E-Class?
    :param year_attr: default on: let's use year as a predictor
    :param price_attr: change between the target log_price and price
    :param feature_selection: model, benchmark, wo_benchmark
    :param model: if the car-model is False (default), we omit this variable from the regression (it would produce
    too many uncontrollable dummies
    :param quantile_percent_upper:
    :param quantile_percent_lower:
    :return:
    """

    if not os.path.exists('statics/pictures'):
        os.makedirs('statics/pictures')

    raw_data = pd.read_csv('data_regressions/1.04.+Real-life+example.csv')
    print(raw_data.head(10))
    print(raw_data.tail(10))

    print(raw_data.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99], include='all'))

    # we see: tops are Volkswagen (936 of 4345), sedan (1649), Diesel (2019), registration yes (3947), E-Class (199)
    # don't mix it up with benchmarks - benchmarks are the one with the highest coeffs./weights

    raw_data.to_excel('data_regressions/used_car_sales_rawdata.xlsx', engine='openpyxl')

    # First observation:
    # BMW sedan vs Mercedes sedan
    # Mercedes crossover for 199,999 - this can't be true! - outlier

    # for purposes later on:
    models_list = raw_data['Model'].unique().tolist()
    print(type(models_list))
    print(models_list)
    print(raw_data['Model'].count())

    brand_list = raw_data['Brand'].unique().tolist()
    engine_list = raw_data['Engine Type'].unique().tolist()
    body_list = raw_data['Body'].unique().tolist()
    registration_list = raw_data['Registration'].unique().tolist()

    print(brand_list)
    print(engine_list)
    print(body_list)
    print(registration_list)

    # ('Model_' + raw_data['Model'].unique())

    df_model = raw_data['Model']
    print(df_model.describe(include='all'))

    # we won't need the Model - too many distinct values, too many dummies:
    data_no_models = raw_data.drop('Model', axis=1)

    # drop missing values (mv):
    print(data_no_models.isnull().sum())
    data_no_mv = data_no_models.dropna(axis=0)
    print(data_no_mv.isnull().sum())
    print(data_no_mv.describe())

    # plot distributions as a subplot-matrix
    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(7, 7))
    fig.suptitle('Comparison of feature-distributions before preprocessing/cutting outliers')
    sns.histplot(data_no_mv['Price'], kde=True, ax=axes[0, 0])
    sns.histplot(data_no_mv['Mileage'], kde=True, ax=axes[0, 1])
    sns.histplot(data_no_mv['EngineV'], kde=True, ax=axes[1, 0])
    sns.histplot(data_no_mv['Year'], kde=True, ax=axes[1, 1])

    plt.savefig("statics/pictures/distributions_features_example_improved_data_no_mv.png", facecolor='white',
                edgecolor="white", dpi=300)
    # plt.show()

    """
    Examine the features and choose percentages where we cut of the outliers - upper and lower cut-offs:
    
    Price: the lower 5% seem reasonable: old cars, high mileage, diversity of brands and models, a lot of unregistered
    cars. Many sedans, hatches and type "other", just little crossover (aka fancy cars which could be expensive)
    The upper 5% appear to have many outliers: Almost only Mercedes and BMW, many crossovers, many cars with 0 or 1 unit
    and coincidentally the same price (seem to be new or semi-new cars)
    Boundary of 6.5 Litre engine volume is not exceeded - so we can later delete the biased rows with unreal engine vol.
    
    Conclusion: we should cut-off the upper 5%-quantile - maybe more, since our goal is to predict a diverse used-car-
    market, it would be okay to cut-off all the Mercedes and BMW which bias our "normal" market. Most models are tuned
    AMG, S-classes or GLS.
    
    Mileage: there are some entries with huge mileages (above 600k) and expensive price respectively - so we can
    delete at least 1% of the upper boundary
    At the lower boundary we observe a lot of new cars with 0 mileage and high prices (Mercedes, BMW) - the same time
    we see different brands with 1k mileage and low and high prices respectively. This looks absolutely not convincing,
    so, do we have errors here like typos, or crash-cars etc.? It' better to cut-off between 1 and 5% of the
    lower quantile.
    
    Engine-Volume: we can cut-off anything above 6.5 litres - those data looks implausible
    the small engines are very affordable and above average Petrol-engines
    
    """
    # calculate 5 % of data for sort-indexer (has to be integer):
    # we count the top 5% of Price, Mileage and Engine-volume; how can we use this knowledge, can we spot some
    # outliers which falsify our general and average prediction?

    print(data_no_mv.sort_values(by=['Price'], ascending=False).head(int((data_no_mv['Brand'].count()*0.05).round(0))))
    print(
        data_no_mv.sort_values(by=['Mileage'], ascending=True).head(int((data_no_mv['Brand'].count()*0.05).round(0))))
    print(
        data_no_mv.sort_values(by=['EngineV'], ascending=True).head(int((data_no_mv['Brand'].count()*0.05).round(0))))

    """
          Brand      Price       Body  Mileage  EngineV Engine Type   
    1728  Mercedes-Benz 300000.000      sedan       68    6.000      Petrol  \
    4318  Mercedes-Benz 300000.000      other       37    5.000      Petrol   
    4101  Mercedes-Benz 295000.000      sedan       29    6.000      Petrol   
    3164  Mercedes-Benz 295000.000      sedan       29    6.000      Petrol   
    2243  Mercedes-Benz 250000.000      other        6    5.500      Petrol   
    2188  Mercedes-Benz 249999.000      other        3    5.500      Petrol   
    2011  Mercedes-Benz 222000.000  crossover        0    6.300      Petrol   
    4044  Mercedes-Benz 219900.000  crossover       30    6.300      Petrol   
    3645  Mercedes-Benz 219900.000  crossover       33    5.500      Petrol   
    3928  Mercedes-Benz 200000.000        van       19    3.500      Petrol   
    3244  Mercedes-Benz 199999.000  crossover        0    5.500      Petrol   
    5     Mercedes-Benz 199999.000  crossover        0    5.500      Petrol   
    602   Mercedes-Benz 199999.000  crossover        0    5.500      Petrol   
    348   Mercedes-Benz 199999.000  crossover        0    5.500      Petrol   
    438   Mercedes-Benz 195000.000  crossover        2    5.500      Petrol   
    131          Toyota 195000.000  crossover        0    4.500      Diesel   
    2521         Toyota 195000.000  crossover        0    4.500      Diesel   
    2657  Mercedes-Benz 189999.000      other        0    4.700      Petrol   
    3636  Mercedes-Benz 187555.000      other        0    4.700      Petrol   
    2550  Mercedes-Benz 185555.000      other        0    4.700      Petrol   
    2130  Mercedes-Benz 177777.000  crossover        0    5.500      Petrol   
    1644  Mercedes-Benz 177000.000  crossover        0    5.500      Petrol   
    993   Mercedes-Benz 173333.000      other        0    3.000      Petrol   
    173   Mercedes-Benz 169000.000      other        1    4.700      Petrol   
    1752  Mercedes-Benz 169000.000        van        1    3.000      Diesel   
    594          Toyota 164500.000  crossover        0    4.500      Diesel   
    2744  Mercedes-Benz 160000.000      other        8    4.600      Petrol   
    3330  Mercedes-Benz 159999.000      other        0    3.000      Petrol   
    1304  Mercedes-Benz 154999.000  crossover        0    4.600      Petrol   
    1658  Mercedes-Benz 144900.000      sedan       16    3.000      Diesel   
    2226  Mercedes-Benz 140000.000  crossover       88    5.460      Petrol   
    1885            BMW 139000.000  crossover        0    4.400      Petrol   
    2212  Mercedes-Benz 138000.000  crossover        0    3.000      Diesel   
    1436            BMW 137000.000  crossover        2    4.400      Petrol   
    1914  Mercedes-Benz 137000.000  crossover       82    5.500      Petrol   
    3167  Mercedes-Benz 135555.000      sedan       54    4.660      Petrol   
    218          Toyota 135000.000  crossover        1    4.500      Diesel   
    62              BMW 133000.000  crossover        7    4.400      Petrol   
    847   Mercedes-Benz 129999.000      sedan       37    4.700      Petrol   
    2209  Mercedes-Benz 129999.000  crossover        1    2.990      Diesel   
    2034  Mercedes-Benz 129999.000  crossover        0    4.670      Petrol   
    3290            BMW 129222.000      sedan        2    5.000      Petrol   
    4264  Mercedes-Benz 126000.000      sedan       14    3.000      Diesel   
    1557  Mercedes-Benz 125000.000      sedan       17    3.000      Diesel   
    2486  Mercedes-Benz 125000.000      sedan       18    3.000      Diesel   
    1346  Mercedes-Benz 125000.000      sedan       18    3.000      Diesel   
    4340  Mercedes-Benz 125000.000      sedan        9    3.000      Diesel   
    3059  Mercedes-Benz 124999.000  crossover        0    3.000      Diesel   
    2393  Mercedes-Benz 124000.000  crossover        0    4.000      Petrol   
    2857  Mercedes-Benz 124000.000  crossover        0    3.000      Petrol   
    ...             ...        ...        ...      ...      ...         ...   
    997          Toyota  72000.000  crossover       40    4.500      Diesel   
    
    Mileage: delete all zeros at least (error or new cars) and maybe all 1's (almost new) - they falsify
    the used-cars market
    All zeros are 185 (4,6% of data_no_mv), all zeros and ones 312 (7,8% data_no_mv) - 
    4025 rows data without missing values.
    The zeros are outside the quantile_percent_lower
    """
    # Todo: cut-off here the lower and upper quantiles of price, mileage and engine volume:
    q_price_lower = data_no_mv['Price'].quantile(quantile_percent_lower)
    q_price_upper = data_no_mv['Price'].quantile(quantile_percent_upper)
    q_mileage_lower = data_no_mv['Mileage'].quantile(quantile_percent_lower)
    q_mileage_upper = data_no_mv['Mileage'].quantile(0.99)
    q_engine_lower = data_no_mv['EngineV'].quantile(quantile_percent_lower)
    q_engine_upper = data_no_mv['EngineV'].quantile(quantile_percent_upper)
    q_year_lower = data_no_mv['Year'].quantile(0.01)

    # upper and lower boundaries of 95% of the data:
    data = {'price_lower': q_price_lower, 'price_upper': q_price_upper, 'mileage_lower': q_mileage_lower,
            'mileage_upper': q_mileage_upper, 'engine_lower': q_engine_lower, 'engine_upper': q_engine_upper}
    q_summary = pd.DataFrame(data=data, index=[0])
    print(q_summary)
    # the following syntax overrides the column data_no_mv:
    # Todo: play with mileage w/o zeros and without lower 5%
    # data_wo_mileage_zeros = data_no_mv[data_no_mv['Mileage'] != 0]
    data_wo_outliers = data_no_mv[data_no_mv['Price'] < q_price_upper]
    data_wo_outliers = data_wo_outliers[data_wo_outliers['Mileage'] < q_mileage_upper]
    data_wo_outliers = data_wo_outliers[data_wo_outliers['Mileage'] > q_mileage_lower]
    data_wo_outliers = data_wo_outliers[data_wo_outliers['EngineV'] < 7]

    # Todo: tinker here with some different quantiles of 'Year':
    #data_wo_outliers = data_wo_outliers[data_wo_outliers['Year'] > q_year_lower]
    data_wo_outliers = data_wo_outliers[data_wo_outliers['Year'] > 1990]

    data_cleaned = data_wo_outliers.reset_index(drop=True)

    # plot distributions as a subplot-matrix after cleansing data
    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(7, 7))
    fig.suptitle('Comparison of feature-distributions after cleansing')
    sns.histplot(data_cleaned['Price'], kde=True, ax=axes[0, 0])
    sns.histplot(data_cleaned['Mileage'], kde=True, ax=axes[0, 1])
    sns.histplot(data_cleaned['EngineV'], kde=True, ax=axes[1, 0])
    sns.histplot(data_cleaned['Year'], kde=True, ax=axes[1, 1])

    plt.savefig("statics/pictures/distributions_features_example_improved_data_wo_outliers.png", facecolor='white',
                edgecolor="white", dpi=300)

    # print(data_no_mv.describe(include='all'))
    data_no_mv.describe(include='all').to_excel('data_regressions/used_car_sales_data_no_mv.xlsx',
                                                engine='openpyxl')
    data_cleaned.describe(include='all').to_excel('data_regressions/used_car_sales_data_preprocessed.xlsx',
                                                engine='openpyxl')

    # Check the OLS assumptions:
    # 1. Check for linearity:
    # let's plot the datapoints of feature-targets-pairs for that we can see the distributions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
    fig.suptitle('Scatter plots of features and targets "Price"')
    ax1.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
    ax1.set_title('Price and Mileage')
    ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
    ax2.set_title('Price and Engine-Volume')
    ax3.scatter(data_cleaned['Year'], data_cleaned['Price'])
    ax3.set_title('Price and Year')

    # we can spot exponential curves with the year, enginev and mileage and have to transform the variables
    data_cleaned['log_price'] = np.log(data_cleaned['Price'])

    # 1. Check for linearity - now with log-price-transformation:
    # let's plot the datapoints of feature-targets-pairs for that we can see the distributions
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
    f.suptitle('Scatter plots of features and targets "log_price"')
    ax1.scatter(data_cleaned['Mileage'], data_cleaned['log_price'], alpha=0.2)
    ax1.set_title('Log-Price and Mileage')
    ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'], alpha=0.2)
    ax2.set_title('Log-Price and EngineV')
    ax3.scatter(data_cleaned['Year'], data_cleaned['log_price'], alpha=0.2)
    ax3.set_title('Log-Price and Year')

    plt.show()

    """
    take it again in if it won't work below:
    """
    data_cleaned_copy_with_all_prices = data_cleaned.copy()
    # Todo: task by instructor: use Price instead of log_price

    match price_attr:
        case 'log_price':
            data_cleaned.drop(['Price'], axis=1, inplace=True)
        case 'price':
            data_cleaned.drop(['log_price'], axis=1, inplace=True)

    # 2. Check for no endogeneity:
    # omitted variable bias: we must not forget a crucial variable otherwise the error-term would enlarge
    # in our example the brand Porsche could explain high prices of small cars with a high mileage
    # so we must not forget the brand as a feature
    # Todo: for homework in a second step when we hone the original model - let's use the "Model"
    # Todo: Result: including models did not add any explanatory power - distinct predictors were too narrow, drop model

    # 3. Check for normality, zero mean and homoscedasticity:
    # homoscedast. holds true since we already log-transformed

    # 4. Check for no autocorrelation - True, we have no time series, just a snapshot of the moment

    # 5. Check for no multicollinearity
    # we check it with the help of statsmodels vif-function (Variance Inflation Factor):

    print(data_cleaned.columns.values)
    # let's check intercorrelation of only the numerical values:
    variables = data_cleaned[['Mileage', 'EngineV', 'Year']]
    vif = pd.DataFrame()
    # List comprehension :-)
    # let's check for the intercorrelation of the variables
    vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif['Features'] = variables.columns
    print(vif)

    # Todo: try to improve by switching year on/off as a predictor:
    match year_attr:
        case 'off':
            # we decide to take out the 'year' due to its high VIF
            data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
        case 'on':
            # IMPROVEMENT: let's play with the year and include it:
            data_no_multicollinearity = data_cleaned

    # create the dummy variables
    # reorder: bringing benchmark into first place
    """
        Convert categorical variable into dummy/indicator variables.
        Each variable is converted in as many 0/1 variables as there are different values. Columns in the output are each 
        named after a value; if the input is a DataFrame, the name of the original variable is prepended to the value.
        :param drop_first=True - we skip the first categorical var. for avoiding multicollinearity
    """
    data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
    # data_with_dummies = pd.get_dummies(data_no_multicollinearity)
    #print(data_with_dummies)

    # the video says "data_with_dummies.columns.values but the output comes without separating commas.
    # so it's better to use only .columns for that we get a list comma-separated and can reuse it for a new variable.
    print('List of all dummies:', data_with_dummies.columns.tolist())
    data_with_dummies_list = data_with_dummies.columns.tolist()

    """
    We need to rearrange a bit and later bring back the "firsts" so that we can use other categorical values as
    dropped-first-dummies.
    """

    cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
            'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
            'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
            'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
            'Registration_yes']

    # improvement 1: benchmarks out - crossover, mercedes, diesel
    cols_wo_benchmarks = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Audi',
            'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
            'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
            'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol', 'Registration_no']

    cols_w_model = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
            'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
            'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
            'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
            'Registration_yes', 'Model_320', 'Model_Sprinter 212', 'Model_S 500', 'Model_Q7', 'Model_Rav 4',
 'Model_A6', 'Model_Megane', 'Model_Golf IV', 'Model_19', 'Model_A6 Allroad',
 'Model_Passat B6', 'Model_Land Cruiser 100', 'Model_Clio', 'Model_318',
 'Model_Polo', 'Model_Outlander', 'Model_A8', 'Model_Touareg', 'Model_Vito',
 'Model_Colt', 'Model_Z4', 'Model_Pajero Wagon', 'Model_X5', 'Model_Caddy',
 'Model_Camry', 'Model_528', 'Model_TT', 'Model_G 55 AMG', 'Model_X6',
 'Model_Galant', 'Model_525', 'Model_Kangoo', 'Model_ML 350', 'Model_730',
 'Model_Trafic', 'Model_S 350', 'Model_Lancer', 'Model_E-Class',
 'Model_Scenic', 'Model_330', 'Model_Passat B5', 'Model_A3',
 'Model_Land Cruiser Prado', 'Model_Caravelle', 'Model_Avensis',
 'Model_GL 320', 'Model_GL 450', 'Model_Lancer X', 'Model_200', 'Model_520',
 'Model_Outlander XL', 'Model_A5', 'Model_X6 M', 'Model_Golf III',
 'Model_A 150', 'Model_FJ Cruiser', 'Model_Koleos', 'Model_Passat B7',
 'Model_Scirocco', 'Model_M5', 'Model_Venza', 'Model_80', 'Model_ML 270',
 'Model_Tiguan', 'Model_C-Class', 'Model_Lupo', 'Model_5 Series']

    # we won't need the Model - too many distinct values, too many dummies:

    data_preprocessed = data_with_dummies   #[cols] or other arrays

    # the following spreadsheet contains either 'price' or 'log_price':
    data_preprocessed.describe(include='all').to_excel('data_regressions/no_models_used_car_sales_data_no_mv.xlsx',
                                                        engine='openpyxl')

    """
        Linear regression model
        Test 1: oops, I commented out data_cleaned.drop(['Price'] in line 360: this 
        leads to an R^2 of .94! Something is wrong with my control flow...no: if I use log_price as target and I
        forget to drop 'price' then in my features I have price as a predictor of log_price/price :-)
        
        Test 2: I dropped log_price in line 360
         R^2 - trainingsdata: 0.7250321231854443
            R^2 - testdata: 0.6855391981940218
        and no plotting of Targets-vs-Predictions - why??? (my x and ylim were constrained :-)
        
        Observation: a 7.71 log-target leads to a 9 log-prediction,  means: a 2,300 Dollar-car is predicted to cost
        more than 8,000 Dollars - that's a massive outlier
        
        Test 3 - Log-price and changing of....:
    """

    match price_attr:
        case 'log_price':
            # Declare the variables
            features = data_preprocessed.drop(['log_price'], axis=1)
            targets = data_preprocessed['log_price']

        case 'price':
            # Declare the variables with Price instead of log_price: (task from the instructor)
            # Todo: task by instructor: use Price instead of log_price - Test 2
            features = data_preprocessed.drop(['Price'], axis=1)
            targets = data_preprocessed['Price']

    # scale the data:
    scaler = StandardScaler()
    scaler.fit(features)
    # it is not usually recommended to standardize all feature-vars. - later we learn how to implement a custom scaler
    features_scaled = scaler.transform(features)

    # test-train-split
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, targets, test_size=0.2, random_state=42)

    # create the linear regression
    reg = LinearRegression()
    reg.fit(x_train, y_train)

    # Let's check graphically, if our first approach of the regression appears already satisfying:
    y_hat = reg.predict(x_train)
    # print(np.exp(y_hat))
    plt.scatter(y_train, y_hat)
    plt.title('Comparison Targets and Predictions of the training data')
    plt.xlabel('Targets (y_train)', size=18)
    plt.ylabel('Predictions (y_hat)', size=18)
    match price_attr:
        case 'log_price':
            plt.xlim(6, 13)
            plt.ylim(6, 13)
        case 'price':
            plt.plot(range(-10000, 70000), range(-10000, 70000), c='orange', linestyle='dashed', linewidth=2, alpha=0.4)

    # Plot the residuals (the differences between targets and predictions, so we use subtraction):
    sns.displot(y_train - y_hat, kde=True, height=7, aspect=1.5)
    plt.title('Residuals PDF: normal, zero mean, but a long left tail', size=18)
    plt.show()

    """
    First observations: the high-price-cars seem plausible; the low-price-cars are not very well predicted - too much 
    variability
    """

    # calculate R2
    print(f"""
            R-squared: Our model is explaining {reg.score(x_train, y_train) * 100} % of the variability of the data
        """)

    reg_summary = pd.DataFrame(features.columns, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary['p-values'] = reg.p
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
            Hint: look at Mercedes :-)

            Note: dummy-weights are ONLY compared to their benchmark and NEVER to the continuous variables.


        """)

    # let's take all categorical data into consideration - create dummies also for body and engine type in the first
    # place, then drop the values with the lowest weight, like Mercedes and so on.

    """
    Testing the model:
    """
    # we start the testing part by finding the predictions:
    y_hat_test = reg.predict(x_test)
    plt.plot(range(0, 14), range(0, 14), c='orange', linestyle='dashed', linewidth=2, alpha=0.4)
    plt.scatter(y_test, y_hat_test, alpha=0.4)
    plt.title('Comparison Targets and Predictions of the test data')
    plt.xlabel('Targets (y_test)', size=18)
    plt.ylabel('Predictions (y_hat_test)', size=18)

    match price_attr:
        case 'log_price':
            plt.xlim(6, 13)
            plt.ylim(6, 13)
        case 'price':
            plt.plot(range(-10000, 70000), range(-10000, 70000), c='orange', linestyle='dashed', linewidth=2, alpha=0.4)
    plt.show()

    # let's manually explore how our model performed in a dataframe-performance df_pf:
    # we see the log-prices and have to reconvert with the exponentials:
    """
    If we use prices - the numbers are getting too large:
    linear_regression_example_improved.py:488: 
    RuntimeWarning: overflow encountered in exp
    df_pf = pd.DataFrame(data=np.exp(y_hat_test), columns=['Predictions'])
    """

    match price_attr:
        case 'log_price':
            df_pf = pd.DataFrame(data=np.exp(y_hat_test), columns=['Predictions'])
            df_pf['Test_Targets'] = np.exp(y_test.reset_index(drop=True))
            df_pf['Residuals'] = df_pf['Predictions'] - df_pf['Test_Targets']
            df_pf['Deviation in %'] = np.absolute(df_pf['Residuals'] / df_pf['Test_Targets'] * 100)
            print(df_pf.sort_values(by=['Deviation in %'], ascending=False).head(30))
        case 'price':
            df_pf = pd.DataFrame(data=y_hat_test, columns=['Predictions'])
            df_pf['Test_Targets'] = y_test.reset_index(drop=True)
            df_pf['Residuals'] = df_pf['Predictions'] - df_pf['Test_Targets']
            df_pf['Deviation in %'] = np.absolute(df_pf['Residuals'] / df_pf['Test_Targets'] * 100)
            print(df_pf.sort_values(by=['Deviation in %'], ascending=False).head(30))

    print(df_pf.describe(include='all'))

    # for improving the model we use feature-selection - after insights - you go back up to the beginning
    # of the script and change the features
    # F-regressions:
    # the output shows two arrays: first, F-statistics, second, p-values
    f_reg_data_no_mv = f_regression(data_no_mv[['Mileage', 'EngineV', 'Year']], data_no_mv['Price'])
    print('F-regression data:', f_reg_data_no_mv)

    """
    Also here change between Price and log_price
    """
    match price_attr:
        case 'log_price':
            f_reg_data_cleaned = f_regression(data_cleaned[['Mileage', 'EngineV', 'Year']],
                                              data_cleaned['log_price'])
        case 'price':
            f_reg_data_cleaned = f_regression(data_cleaned[['Mileage', 'EngineV', 'Year']],
                                              data_cleaned['Price'])

    print('F-regression data cleaned:', f_reg_data_cleaned)

    p_values = f_reg_data_cleaned[1]
    print(f'The p-values for X are: {p_values.round(3)}')

    print('Doublecheck with the p-parameter from the overridden class LinearRegression (we do not need f-regression):')
    print(reg.p)

    r2_train = reg.score(x_train, y_train)
    r2_test = reg.score(x_test, y_test)

    print(f"""
            R^2 - trainingsdata: {r2_train}
            R^2 - testdata: {r2_test}
        """)

    def adjusted_r2(X, y, regression):
        r2 = regression.score(X, y)
        n = X.shape[0]
        p = X.shape[1]
        rsq_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return rsq_adj

    adjr2_train = adjusted_r2(x_train, y_train, reg)
    adjr2_test = adjusted_r2(x_test, y_test, reg)

    print(f"""
            AdjR^2 - trainingsdata: {adjr2_train}
            AdjR^2 - testdata: {adjr2_test}
           """)

    #print(f"""
    #    For testing:
    #    {data_cleaned_copy_with_all_prices}
    #""")

    # for tinkering:
    data_wo_mileage_zeros = data_no_mv[data_no_mv['Mileage'] != 0]
    # print(data_wo_mileage_zeros.sort_values(by='Mileage', ascending=True))
    data_wo_mileage_zeros.to_excel('data_regressions/used_car_sales_data_wo_mileage_zeros.xlsx', engine='openpyxl')

    print(f"""
        Was passiert, wenn ich y_hat mit realem 'price' und nicht 'log_price' nehme:
        Bei log_price sehe ich wieder die exp. echten Preise.
        Bei price ....[inf inf inf ... inf inf inf] wegen overflow encountered in exp...
        {np.exp(y_hat)}
    """)

    # include='all' makes that I also see the dummies :-) Don't forget it!
    print(features.describe(include='all'))


if __name__ == '__main__':
    linear_regression_practical_example_improved(quantile_percent_upper=0.95, quantile_percent_lower=0.05, price_attr=
                                                 'log_price', year_attr='on')


"""
Reasoning, why we let the 'Year' in the model:

Hi Joe,

thanks for your response! Your results look really nice!

1 question though: You are saying that Year and Mileage are the biggest drivers of a second-hand car, 
which obviously makes a lot of sense! But how do you deal with the colinearity problem of this, 
given that the VIF for year was ~ 10?


Thanks

Sunil
JD
Joe
4 Upvotes
Vor 2 Jahren

unfortunately this is an area without very clearly defined rules and the answer to a lot of the questions 
are "it depends".

you have a problem when you can use one independent variable to predict the value of another. 
in the case of mileage and year they are going to be closely related as older cars generally tend 
to have higher mileage, but you will also find newer cars with high mileage 
(which will be worth less than newer cars with low mileage) and older cars with very low mileage 
(which will still be cheap because they're old). if you can't reliably predict the year from the mileage 
then you shouldn't be throwing it out of the model.

this is just one of those situations where knowing how used cars get priced and the way that these variables 
interact in the real world, helped me to decide to ignore the high VIF for year.
"""