"""
README:
Mit diesem Skript werden Daten für eine logistische Regression vorprozessiert und als neue .csv gespeichert.
Danach muss die erste Funktion nicht mehr verwendet werden.

Die Funktion absenteeism_ml_logistic_regression() war ein erster Test und kann weiter zum probieren und
debuggen als "Spielwiese" verwendet werden. Sie funktioniert, wird aber keinen Scaler und kein Regressions-Modell (Reg)
persistent schreiben/speichern.

Die Funktion absenteeism_ml_logistic_regression_improved_custom_scaler() nutzt die Klasse CustomScaler und wird als
ML-Funktion verwendet, um das Modell (reg) und die standardisierten Features (absenteeism_scaler) zu erstellen
und speichern.

Diese Dateien werden später im absenteeism_module_final und/oder in absenteeism_exercise_integration verwendet.

"""


import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt

import openpyxl
from openpyxl import writer

import pickle

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
    'max_rows': 20,
    'precision': 4,
    'float_format': lambda x: f'{x:.2f}'
    }

for option, value in settings.items():
    pd.set_option(f'display.{option}', value)


def absenteeism_preprocessing():
    """
    A popular tool in data analytics, machine learning, advanced statistics, and econometrics, is regression analysis.

    Roughly speaking, this is an equation which on one side has a variable, called a dependent variable,
    because its value will depend on the values of all variables you see on the other side.

    The variables on the right side are all independent, or explanatory.
    Their role is to explain the value of the dependent variable.

    There are more terms that can be used for these variables in the same context.
    The dependent variable can also be called a target, while the independent variables can be called predictors.

    Furthermore, as mentioned in the previous video, to avoid confusion with the term ‘variable’,
    which has a different meaning when used by programmers, we, as BI analysts or data scientists,
    will call the explanatory variables ‘features’. Remember that if you prefer,
    you could call them ‘attributes’ or ‘inputs’ as well.

    Logistic regression:
    It is a type of regression model whose dependent variable is binary.
    That is, the latter can assume one of two values – 0 or 1, True or False, Yes or No.

    Therefore, considering the values of all our features, we want to be able to predict whether
    the dependent variable will take the value of 0 or 1.

    Feature descriptions:
    reasons: Leute haben einen Krankenschein abgegeben. Reasons 1 - 21 are registered in the international
    classification of diseases (ICD), Reasons 22 - 28 are not.
    Date: date of absence
    Transportation expense: costs related to business travel such as fuel, parking and meals, in our table:
    monthly transportation expenses of an individual measured in dollars
    Distance to work: measured in kilometres - the distance an individual must travel from home to work
    Daily work load average: measured in minutes - the average amount of time spent working per day
    Absenteeism time in hours

    :return:
    """
    raw_data = pd.read_csv('data_csv_excel/Absenteeism_data.csv')
    #print(raw_data)
    print(raw_data.describe(include='all'))

    df = raw_data.copy()

    # checking for missing values - no missing values here; very good:
    print(df.info())

    df.drop(columns=['ID'], inplace=True)   # or: drop(['ID], axis=1, inplace=True)
    print(df)

    df.rename(columns={'Reason for Absence': 'reason'}, inplace=True)

    print('Python list comma separated:', df['reason'].unique().tolist())
    print('Pandas array in my IDE without commas:', pd.unique(df['reason']))
    print('How many distinct values - one is missing:', len(df['reason'].unique()))
    print(sorted(df['reason'].unique()))

    """ Get Dummies 
    Exkurs: Why we drop the first dummy-column? Why we always only use n-1-dummies?
    
    Do you include all dummy variables in a regression model?

    If you have 3 groups for race, then you can use only 2 dummy variables to represent membership in race group.
    
    In general, for k groups, you use only (k-1) dummy variables.
    
    It’s helpful to think of each dummy variable as a yes/ no question about group membership.
    
    Suppose your race groups are:
    
    1= white
    
    2 = black
    
    3 = other
    
    Dummy variable 1 answers the question: Do you identify yourself as white? 0 = no, 1 = yes.
    
    Dummy variable 2 answers the question: Do you identify yourself as black? 0 = no, 1 = yes.
    
    Provided that your groups are mutually exclusive and exhaustive, then if a person answers no to the first 
    two questions, that person must be a member of group 3, other race.
    
    In fact, if you try to include a third dummy variable in this situation, regression analysis will fail 
    because the scores on the third dummy variable are perfectly predictable from the answers 
    on the first two dummy variable questions.
    
    Assume we have 3 categories, which are mutually exclusive and collectively exhaustive (MECE). And let’s have three dummies: d1, d2, and d3

    . They can only take values 0 and 1.
    
    Due to this MECE condition, d1+d2+d3=1
    
    by definition.
    
    Now, take a linear model explained only by these dummies.
    
    y=b0+b1d1+b2d2+b3d3
    
    Since the problem is MECE, for each case we take, we would have one of these situations:
    d1=d2=0, d3=1
    
    d1=d3=0, d2=1
    
    d2=d3=0, d1=1
    
    Now, if you put that in a matrix,
    
    Now, if you put that in a matrix,
    
    ⎛⎝⎜y1y2y3⎞⎠⎟=B⎛⎝⎜111    d11d21d31   d12d22d32   d13d23d33⎞⎠⎟
    
    where B is some coefficients matrix, which is of no interest. The column of 1s signifies the constant (b0
    
    ), and the next 3 columns are the values of the first dummy, the second dummy, and the third dummy 
    for each of the three observations.
    
    Now, if we take the above case and substitute, we get:
    
    ⎛⎝⎜y1y2y3⎞⎠⎟=B⎛⎝⎜111    100 010 001⎞⎠⎟
    
    The rank of this matrix seems to be 4, right?
    
    The big issue here, however, is that the problem was MECE, so we also know that d1+d2+d3=1
    
    But we have infitely many solutions to this problem (we can’t define the dummies).
    
    One of these columns is useless as it is linearly dependent on the others, given the MECE condition.
    
    Since we have perfect linear dependence (perfect multicollinearity, sorry about that), 
    we get rid of one of the dummy columns and are left with two dummies.
    
    When two of the dummies are 0, then the third one is 1 for sure, so you get rid of it.
    
    Generalize this result and you get that for n categories, you need n-1 dummies.
    
    Reasoning: die erste Spalte (erster Dummy) wird von allen anderen Spalten erklärt - entweder/oder,
    wenn es die anderen Dummys nicht sind, dann ist es der erste Dummy - wir hätten Multicollinearity
    Also, können wir die erste Spalte (hier im Beispiel Grund 0, also "unentschuldigt", gleich weglassen.
    """
    # get dummies out off the categorical nominal reasons and sort the 29 reasons (mostly diseases)
    # into reasonable groups (some diseases are akin to each other - a separation would not provide any more
    # predictive power

    df_withdummies = pd.get_dummies(df['reason'], drop_first=True)

    # check for missing values within the dummies: 0 means missing value (there is no reason aka 0), more than
    # 1 would be a problem since there must be only one reason for the absenteeism
    df_withdummies['reasons_check'] = df_withdummies.sum(axis=1)
    #print(df_withdummies)
    df_withdummies.drop(['reasons_check'], axis=1, inplace=True)

    """ Group the reasons for absence: it's a kind of classification...
    
    Dealing with 28 reasons for absence and thus more than 40 columns would end up in too much wasted computational
    power - so it is proper to group/classify the reasons along some mutual characteristics like diseases and
    light reasons like a patient-follow-up or a dentist-appointment...1 - 14 diseases, 15 - 17 pregnancy,
    18 - 21 poisoning and other signs not related to anywher else, 22 - 28 light reasons
    """
    #Todo: den gleichen qualitativen Ansatz für mein Gebrauchtwagenbeispiel verwenden.

    """ More hacks for checking DFs"""
    print(df.columns.values)
    print(df_withdummies.columns.values)
    # so we can drop the original reason-column:
    # print(df['reason'])
    df.drop(['reason'], axis=1,  inplace=True)  # axis 1 is always horizontal

    # create new dataframes... loc[first colon means all rows, then use numbers of cols OR strings of col-names (labels)
    # since every observation (row) only got one reason for absence, you use the trick with the max-method
    # to group the columns per reasons_group - as a result you get one column per group :-)
    reasons_type_1 = df_withdummies.loc[:, 1:14].max(axis=1)
    reasons_type_2 = df_withdummies.loc[:, 15:17].max(axis=1)
    reasons_type_3 = df_withdummies.loc[:, 18:21].max(axis=1)
    reasons_type_4 = df_withdummies.loc[:, 22:].max(axis=1)
    print(reasons_type_1.shape)     # proof for only one column

    """ Convert the day - slashes indicates that we have strings and need to convert into datetime """
    # check it baby...
    print('Will show us "string":', type(df['Date'][0]))

    # we apply a timestamp, always specify the proper format...we give python the current state including
    # the slashes for knowing how to read the current format...
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")

    df['day_of_week'] = df['Date'].dt.dayofweek

    #days = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    #df['day_of_week_names'] = df['day_of_week'].apply(lambda x: days[x])

    # I don't know where the dt comes from... but it works...
    df['Month_value'] = df['Date'].dt.month

    """
    The solution from the course:
    list_months = []
    for i in range(df.shape[0]):
        list_months.append(df['Date'][i].month)
    
    df['Month_value'] = list_months
    
    Weekday - another alternative - weekday()-method:
    #print(df['Date'][300].weekday())
    
    Lesson weekday - to apply a certain type of modification iteratively on a each value from a series or a column
    in a DataFrame, it is great idea to create a function that can execute this operation for one element,  and then
    implement it to all values from the column of interest:
    
    1. Create a mini-function (we also can use Lambda-functions)
    def date_to_weekday(date_value):
        return date_value.weekday()
        
    2. We use apply() to apply our mini-function:
    df['day_of_week'] = df['Date'].apply(date_to_weekday)
     
    """

    df.drop(['Date'], axis=1, inplace=True)
    df_converted_dates = df.copy()

    """ Working on education, children, pets - categorical data containing integers: """

    print(df_converted_dates['Education'].unique().tolist())
    education_mapping = {1: 'high_school',
                         2: 'graduate',
                         3: 'postgraduate',
                         4: 'master_or_doctor'}

    # Mega Funktion value_counts() - besser als unique
    print(df_converted_dates['Education'].value_counts())
    # ursprünglich hatte ich die Column "education_binary" genannt - overriden ist besser, damit man später
    # nicht vergisst zu löschen
    df_converted_dates['Education'] = df_converted_dates['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

    # concatenate all partial dataframes...
    joined_frames_list = [df_converted_dates, reasons_type_1, reasons_type_2, reasons_type_3, reasons_type_4]
    main_df = pd.concat(joined_frames_list, axis=1, join='outer')
    # shorter alternative:
    # main_df = pd.concat([df_converted_dates, reasons_type_1, reasons_type_2, reasons_type_3, reasons_type_4],
    # axis=1, join='outer')

    # rename all columns we would like to alter:
    # print(main_df.columns.values)

    # coole variante, um mit einer Liste durch neues assignen alle columns umzubenennen :-)
    column_names = ['Transportation Expense', 'Distance to Work', 'Age',
                    'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets',
                    'Absenteeism Time in Hours', 'day_of_week', 'Month_value',
                    'reason_1', 'reason_2', 'reason_3', 'reason_4']
    main_df.columns = column_names
    # Alternative:
    #main_df.rename(columns={0: 'group1', 1: 'group2', 2: 'group3', 3: 'group4'}, inplace=True)

    """   Reorder columns"""
    # puny little task... :-)
    # einfach durch erstellen eines neuen df ;-)
    column_names_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'Month_value', 'day_of_week',
                              'Transportation Expense', 'Distance to Work', 'Age',
                              'Daily Work Load Average', 'Body Mass Index', 'Education',
                              'Children', 'Pets', 'Absenteeism Time in Hours']

    main_df = main_df[column_names_reordered]
    print(main_df)

    # create a checkpoint all way long...
    # Todo: also do this in your financial data df for avoiding fragmentation
    df_preprocessed = main_df.copy()
    print(df_preprocessed.info())



    """ Plotting """
    # check for linearity:

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(16, 4))
    fig.suptitle('Abwesenheit in Abhängigkeit diverser Features')
    ax1.scatter(df['Daily Work Load Average'], df['Absenteeism Time in Hours'])
    ax1.set_title('Daily Workload')
    ax2.scatter(df['Education'], df['Absenteeism Time in Hours'])
    ax2.set_title('Education')
    ax3.scatter(df['Transportation Expense'], df['Absenteeism Time in Hours'])
    ax3.set_title('Transportation $')
    ax4.scatter(df['Age'], df['Absenteeism Time in Hours'])
    ax4.set_title('Age')
    plt.show()

    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(7, 7))
    fig.suptitle('Comparison of feature-distributions before preprocessing/cutting outliers')
    sns.histplot(df['Children'], kde=True, ax=axes[0, 0])
    sns.histplot(df['Distance to Work'], ax=axes[0, 1])
    sns.histplot(df['Transportation Expense'], kde=True, ax=axes[1, 0])
    sns.histplot(df['Daily Work Load Average'], kde=True, ax=axes[1, 1])
    plt.show()

    sns.histplot(df['Children'], binrange=(0, 5), binwidth=1)
    plt.title('Children Histogram')
    plt.show()

    df_preprocessed.to_csv('data_csv_excel/df_preprocessed.csv', index=False)
    df_preprocessed.to_excel('data_csv_excel/Absenteeism_df_preprocessed.xlsx', engine='openpyxl')


def absenteeism_ml_logistic_regression():
    """
    Define our targets y: what is excessive absenteeism (1) and what is moderate absenteeism (0)?
    We will use the median value of the Absenteeism time in hours as a cut-off line

    Choose wisely the starting predictors: reasons 1 - 4, work load (the rationale in the course is: the busier
    a person is, the less she/he will skip work), children, pets, distance from home - since you need more time
    when you visit the doc with you child for instance.

    The model itself will give us a fair indication of which variables are important for the analysis.
    :return:
    """
    df_preprocessed = pd.read_csv('data_csv_excel/df_preprocessed.csv')
    df_preprocessed.rename(columns={'Absenteeism Time in Hours': 'absent_time'}, inplace=True)
    print(df_preprocessed)

    """ Create the targets... """
    # classify employees into classes -
    # map absent_time to 1 and 0 with np.where (that's like if in Excel):
    # new method for mapping using np.where():
    median = df_preprocessed['absent_time'].median(axis=0)
    targets_alt = np.where(df_preprocessed['absent_time'] <= median, 0, 1)
    print(targets_alt, sum(targets_alt))

    targets = np.where(df_preprocessed['absent_time'] > median, 1, 0)
    print(targets, sum(targets), '-> same result')

    df_preprocessed['excessive_absenteeism'] = targets
    #print(df_preprocessed)

    """
    A comment on the targets:
    using the median as naive cut-off line is rigid, numerical stable and implicitly balanced since by the nature
    of the median we get around 50% of 0s and 1s. That's important since we otherwise would get an algorithm which
    only responds 0s or 1s.
    """

    proof_of_balance = targets.sum() / targets.shape[0]
    print(f'Proof of balance: {proof_of_balance.round(2)}')

    # checkpoint:
    data_with_targets = df_preprocessed.drop(['absent_time'], axis=1)
    print(data_with_targets)
    # check if we have two instances aka a real checkpoint:
    print(data_with_targets is df_preprocessed)

    """ Selecting the inputs (features) """
    # use Numpy library index_tricks - np.r_:
    unscaled_inputs = data_with_targets.iloc[:, np.r_[0:6, 7:12, 13:16]]
    print(unscaled_inputs)

    # scale/standardize them...
    scaler = StandardScaler()
    """
    The scaler-object will be used to subtract the mean and divide by the standard deviation variablewise (featurewise)
    """
    # this line calculates and stores the mean and st.dev.
    scaler.fit(unscaled_inputs)
    # this line subtracts, divides and stores
    scaled_inputs = scaler.transform(unscaled_inputs)
    print(scaled_inputs.shape)

    # train-test-split and shuffle (fixate your one-time-shuffling by the random_state...)
    X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=42)

    """ Training the model """
    reg = LogisticRegression()
    reg.fit(X_train, y_train)

    """ Manually check the accuracy """
    model_outputs = reg.predict(X_train)
    print(type(model_outputs))
    print(model_outputs, f'Anzahl: {model_outputs.shape[0]}')
    print(y_train, f'Anzahl: {y_train.shape[0]}')
    accuracy_manually = np.sum(model_outputs == y_train) / model_outputs.shape[0]
    print(f'accuracy_manually: {accuracy_manually}')

    """ Interpreting the Coefficients/Weights - IMPORTANT NOTE """

    df_summary_table = pd.DataFrame(columns=['Feature name'], data=unscaled_inputs.columns.values)
    df_summary_table['Coefficient_Weight'] = reg.coef_.T    # or np.transpose()

    # Hammer-Trick, um weitere Zeilen an beliebige Stellen einzufügen:
    # we shift the whole index...
    df_summary_table.index = df_summary_table.index + 1
    df_summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    df_summary_table.loc[15] = ['Accuracy', reg.score(X_train, y_train)]
    df_summary_table.sort_index(inplace=True)
    # print(df_summary_table)

    """
        Standardized coefficients are basically the coefficients of a regression where ALL variables
        have been standardized. 

        Whenever we are dealing with a logistic regression, the coefficients we are predicting are the so
        called log-odds. These log-odds are later transformed into 0s and 1s. 

        Let's find the exponentials of these log-odds - new column "odds_ratios"
        Most important (8.49) are at the top
        Interpretation: if its coefficient is around 0 or if its odds-ratio is close to 1, this means
        that the feature is not particularly important (a ratio of 1 changes 1 unit by times 1, thus no change)
        
        For a unit change in the standardized feature, the odds increase by a multiple equal to the odds-ratio.
        
        Hence, we now spot some features which seem to have no merit, like Daily Work Load Average, Distance to Work
        and Day of the week. (For now we let them in the model)

        Given all features, these seem to be the ones that make no difference.
    """

    df_summary_table['odds_ratios'] = np.exp(df_summary_table.Coefficient_Weight)
    df_summary_table.sort_values(by='odds_ratios', ascending=False, inplace=True)

    """
        What about the Reason_0 (no reason for absence)? We dropped it when we were creating the dummies.
        From the coefficients we see, whenever a person stated a certain reason, we have a much higher chance
        to observer excessive absenteeism. A good question would be: How much bigger of a chance?
        
        Big, big problem!
        When we standardized the inputs, we also standardized the dummies. This is bad practice, since when we
        standardized we lose the whole interpretability of a dummy.
        If we would have let them 0s and 1s we could have said e. g. for a unit change it's 8.5 times higher 
        of a chance (reason_1) for being excessively absent.
        
        Our over all accuracy is still valid but we don't know which one of the reasons got the biggest influence
        and highest changes.
        So, we need to correct our model at a checkpoint - we jump back to unscaled_inputs...
        
        We implement a custom-scaler, which will only scale columns we choose on purpose. We will preserve the 
        dummies untouched. See the improved model in the next function...
        
    """




    print(df_summary_table)


    predictions = reg.predict_proba(X_test)
    # print(predictions)

    status = 'You gotta do what you got to do.'
    return status


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Ziel: wir wollen nur numerische Features unseres Dataframes skalieren/standardisieren - also eine Auswahl
    als Liste treffen, die wir dann an diese Klasse übergeben. Die Klasse nutzt aber die geerbten Eigenschaften
    des SKlearn-Standardscalers, also wie der Mittelwert und die Standardabweichung berechnet wird.
    """
    def __init__(self, columns):
        """
        Wir konstruieren hier wie üblich ein Objekt "scaler" des StandardScalers
        Wir definieren gleich eine Variable "columns" es ist eigentlich nur ein extra Argument zusätzlich
        zum StandardScaler...
        :param columns:
        :param copy:
        :param with_mean:
        :param with_std:
        """
        self.scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        """
        Wie beim Standardscaler folgen nun zwei Funktionen fit() und transform() - wir packen diese sonst standalone-
        Methoden einfach nur hier in die Klasse (verkapseln sie) - als X übergeben wir unsere Variable "columns",
        die wir später "zurechtslicen" werden.
        :param X:
        :param y:
        :return:
        """
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        """
        Hier konstruieren wir einen Dataframe, der aus den standardisierten Features und den Dummies besteht.
        Dazu nutzen wir concat(). Der Standardscaler würde alles skalieren und einen mehrdimensionalen Array ausgeben.
        Der Trick hier ist, diesen Array mit dem unskalierten DF zu verknüpfen.
        :param X:
        :param y:
        :param copy:
        :return:
        """
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


def absenteeism_ml_logistic_regression_improved_custom_scaler():
    df_preprocessed = pd.read_csv('data_csv_excel/df_preprocessed.csv')
    #df_preprocessed.rename(columns={'Absenteeism Time in Hours': 'absent_time'}, inplace=True)

    targets = np.where(df_preprocessed['Absenteeism Time in Hours'] >
                       df_preprocessed['Absenteeism Time in Hours'].median(axis=0), 1, 0)
    df_preprocessed['Excessive Absenteeism'] = targets
    #print(df_preprocessed)

    # checkpoint:
    data_with_targets = df_preprocessed.drop(['Absenteeism Time in Hours'], axis=1)
    print(data_with_targets)
    # check if we have two instances aka a real checkpoint:
    print(data_with_targets is df_preprocessed)

    """ Selecting the inputs (features) """
    # use Numpy library index_tricks - np.r_:
    # for backward-elimination: use different slices for features which you can omit
    # 5 is day_of_week, 8 is distance, 10 is daily work load,
    # unscaled_inputs = data_with_targets.iloc[:, np.r_[0:5, 7:8, 9:10, 11:12, 13:16]]
    unscaled_inputs = data_with_targets.iloc[:, :-1]
    print(unscaled_inputs)

    # scale/standardize them with our custom scaler...
    print(unscaled_inputs.columns)  #immer OHNE .values - sonst bekommen wir die Kommas nicht mit
    #columns_to_scale = ['Month_value', 'day_of_week', 'Transportation Expense', 'Distance to Work', 'Age',
                        #'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pets']

    # better ... instead of columns_to_scale we do it reverse ... columns_to_omit and list-comprehension:

    # TODO !!!!!!!!!!!!!!!! Hier muss ich ['day_of_week', 'Daily Work Load Average', 'Distance to Work'] droppen!
    #unscaled_inputs_dropped = unscaled_inputs.drop(['day_of_week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
    columns_to_omit = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'Education']
    columns_to_scale = [entry for entry in unscaled_inputs.columns.values if entry not in columns_to_omit]

    absenteeism_scaler = CustomScaler(columns_to_scale)
    absenteeism_scaler.fit(unscaled_inputs) # die Klasse gleicht durch die isin-Methode mit der Liste ab

    scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
    print(scaled_inputs)

    # meine Variante für backward elimination - not preferable...
    # alt_inputs = scaled_inputs.drop(['Month_value', 'Daily Work Load Average', 'Distance to Work', 'day_of_week'],
                                    #axis=1)

    # train-test-split and shuffle (fixate your one-time-shuffling by the random_state...)
    X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=42)

    """ Train the model """
    reg = LogisticRegression()
    reg.fit(X_train, y_train)

    """ Manually check the accuracy """
    model_outputs = reg.predict(X_train)
    print(type(model_outputs))
    print(model_outputs, f'Anzahl: {model_outputs.shape[0]}')
    print(y_train, f'Anzahl: {y_train.shape[0]}')
    accuracy_manually = np.sum(model_outputs == y_train) / model_outputs.shape[0]
    print(f'accuracy_manually: {accuracy_manually}')

    """ Interpreting the Coefficients/Weights - IMPORTANT NOTE """

    df_summary_table = pd.DataFrame(columns=['Feature name'], data=unscaled_inputs.columns.values)
    df_summary_table['Coefficient_Weight'] = reg.coef_.T  # or np.transpose()

    # Hammer-Trick, um weitere Zeilen an beliebige Stellen einzufügen:
    # we shift the whole index...
    df_summary_table.index = df_summary_table.index + 1
    df_summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    df_summary_table.loc[15] = ['Accuracy', reg.score(X_train, y_train)]
    df_summary_table.sort_index(inplace=True)
    # print(df_summary_table)

    df_summary_table['odds_ratios'] = np.exp(df_summary_table.Coefficient_Weight)
    df_summary_table.sort_values(by='odds_ratios', ascending=False, inplace=True)

    print(df_summary_table)

    """ Backward elimination - we won't need features with weights around 0 or an odds-ratio around 1 
        How to simplify our model...
    """
    """
    We see: our unscaled dummies weigh in much more (reason 3: poisoning and peculiar diseases is accountable
    for excessive absenteeism
    
    Month_value, Daily work load, distance and day of the week hardly influence our model - we can omit them and
    create an alternative inputs-variable above
    """

    """ Testing the model - predictions are yhat-test """

    test_accuracy = reg.score(X_test, y_test)
    print(test_accuracy)

    # very useful method: predict_proba() - returns the probabilitiy estimates for all possible outputs (classes)

    predicted_proba = reg.predict_proba(X_test)

    # first col: probability of being 0, second col: prob of being 1 for all observations of the test-set
    #print(predicted_proba)
    predicted_proba_only_1 = predicted_proba[:, 1]
    print(predicted_proba_only_1)

    # Todo for homework: end of ML-course; do something useful with the predicted_proba_only_1...

    """
        IMPORTANT !!!
    """

    """ Save the model - by pickling 
        We will save the reg-object in a file
    """

    if not os.path.exists('data_models'):
        os.makedirs('data_models')

    with open('data_models/absenteeism_model.pickle', 'wb') as file:
        pickle.dump(reg, file)

    with open('data_models/absenteeism_scaler.pickle', 'wb') as file:
        pickle.dump(absenteeism_scaler, file)

    """ How to deploy the model... """
    # saving the model and apply it to new data...
    # clever approach: Storing code in a module:
    # Storing code in a module will allow us to reuse it without trouble

    status = 'You gotta do what you got to do.'
    # print(alt_inputs.columns)
    print(absenteeism_scaler)
    return status


if __name__ == '__main__':
    # absenteeism_preprocessing()
    # print(absenteeism_ml_logistic_regression())
    print(absenteeism_ml_logistic_regression_improved_custom_scaler())
