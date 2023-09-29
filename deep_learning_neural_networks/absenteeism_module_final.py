import os
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import pickle

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

"""
These are some Object-oriented-classes packed in a module for effortless reusing in machine learning.

We already scaled the data (absenteeism_scaler.pickle) and created a model (absenteeism_model.pickle)
These classes use those files an do some classification with new data.
"""


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        """
        The isin-Method:
        # importing pandas package
        import pandas as pd

        # making data frame from csv file
        data = pd.read_csv("employees.csv")

        # creating filters of bool series from isin()
        filter1 = data["Gender"].isin(["Female"])
        filter2 = data["Team"].isin(["Engineering", "Distribution", "Finance" ])

        # displaying data with both filter applied and mandatory
        data[filter1 & filter2]

        TIPP! The tilde-sign negates isin:
        s[~s.isin(x)]

        1. df[-df["column"].isin(["value"])]
        2. df[~df["column"].isin(["value"])]
        3. df[df["column"].isin(["value"]) == False]
        4. df[np.logical_not(df["column"].isin(["value"]))]

        Note: for option 4 for you'll need to import numpy as np

        Update: You can also use the .query method for this too. This allows for method chaining:
        5. df.query("column not in @values").
        where values is a list of the values that you don't want to include.

        :param X:
        :param y:
        :param copy:
        :return:
        """
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


""" Create the special class that we are going to use from here on to predict new data """


class AbsenteeismModel:
    def __init__(self, model_file, scaler_file):
        """
        read the 'model' and 'scaler' files which were saved
        :param model_file: by the logistic regression we created our model and saved it to a pickle-file
        :param scaler_file: by the logistic regression we created our scaler and saved it to a pickle-file
        """
        with open(model_file, 'rb') as model_file, \
             open(scaler_file, 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            self.df_with_predictions = None
            self.preprocessed_data = None

    def load_and_clean_data(self, data_file):
        """
        take a data file (*.csv) and preprocess it in the same way as in the lectures

        That's the preprocessing step, for the regression-step we don't need a function in this module,
        since we already have the model and scaler.
        :param data_file:
        :return:
        """
        df = pd.read_csv(data_file, delimiter=',')

        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()

        df.drop(['ID'], axis=1, inplace=True)

        # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
        df['Absenteeism Time in Hours'] = 'NaN'

        # create a separate dataframe, containing dummy values for ALL avaiable reasons
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

        # split reason_columns into 4 types
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        # to avoid multicollinearity, drop the 'Reason for Absence' column from df
        df.drop(['Reason for Absence'], axis=1, inplace=True)

        # concatenate df and the 4 types of reason for absence
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

        # assign names to the 4 reason type columns
        # note: there is a more universal version of this code, however the following will best suit
        # our current purposes
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                        'Pets', 'Absenteeism Time in Hours', 'reason_1', 'reason_2', 'reason_3', 'reason_4']
        df.columns = column_names

        # re-order the columns in df
        column_names_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'Date', 'Transportation Expense',
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]

        # convert the 'Date' column into datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # create a list with month values retrieved from the 'Date' column
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)

        # insert the values in a new column in df, called 'Month Value'
        df['Month_value'] = list_months

        # create a new feature called 'Day of the Week'
        df['day_of_week'] = df['Date'].apply(lambda x: x.weekday())

        # drop the 'Date' column from df
        df.drop(['Date'], axis=1, inplace=True)

        # re-order the columns in df
        column_names_upd = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'Month_value', 'day_of_week',
                            'Transportation Expense', 'Distance to Work', 'Age',
                            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                            'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]

        # map 'Education' variables; the result is a dummy
        df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

        # replace the NaN values
        df.fillna(value=0, inplace=True)

        # drop the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'], axis=1)

        # drop the variables we decide we don't need
        # NICHT DROPPEN - sie sind im custom scaler!!!!
        # im Video sind sie gedroppt - Ergebnis bleibt aber fast gleich...
        #df = df.drop(['day_of_week', 'Daily Work Load Average', 'Distance to Work'], axis=1)

        # we have included this line of code if you want to call the 'preprocessed data'
        self.preprocessed_data = df.copy()

        # we need this line so we can use it in the next functions
        self.data = self.scaler.transform(df)

    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if self.data is not None:
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data

    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data


if __name__ == '__main__':
    model = AbsenteeismModel(model_file='data_models/absenteeism_model.pickle',
                             scaler_file='data_models/absenteeism_scaler.pickle')
    model.load_and_clean_data(data_file='data_models/Absenteeism_new_data.csv')
    print(model.predicted_outputs())

