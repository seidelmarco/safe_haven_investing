import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
    'max_rows': 50,
    'precision': 2,
    'float_format': lambda x: f'{x:.3f}'
    }

for option, value in settings.items():
    pd.set_option("display.{}".format(option), value)


def confusion_matrix(data, actual_values, model):
    """
    Confusion matrix
    :param data: data frame or array, data is a data frame formatted in the same way
    as your input data (without the actual values) e.g. const, var1, var2, etc. Order is very important!
    :param actual_values: data frame or array, These are the actual values from the test_data.
    In the case of a logistic regression, it should be a single column with 0s and 1s
    :param model: a LogitResults object - this is the variable where you have the fitted model
    e.g. results_log in this course
    :return:
    """

    # Predict the values using the Logit model
    pred_values = model.predict(data)

    # Specify the bins
    bins = np.array([0, 0.5, 1])

    # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
    # if they are between 0.5 and 1, they will be considered 1
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

    # Calculate the accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

    # Return the confusion matrix and accuracy
    return cm, accuracy

    # Fix: split the data into training and testing. We create the regression on the training data - after
    # we have the coeffs we test the model on the test data by creating a confusion matrix and assessing the
    # accuracy. We strive for high training and test accuracy!


def logistic_regression_basics():
    """

    :return:
    """
    raw_data = pd.read_csv('data_regressions/2.01.+Admittance.csv')
    print(raw_data)
    data = raw_data.copy()
    data['Admitted'] = raw_data['Admitted'].map({'Yes': 1, 'No': 0})
    print(data)

    X1 = data['SAT']
    y = data['Admitted']

    X = sm.add_constant(X1)

    # let's see if a linear regression makes sense:
    reg_lin = sm.OLS(y, X).fit()
    print(reg_lin.summary())

    plt.scatter(X1, y, c='C0', alpha=0.3)

    # in our first training-example we printed the summary, read the params and hard-coded the formula/model
    # now we parametrize it by using the instance params of the Result-class: 0 is Intercept/Bias, 1 is the weight
    y_hat = X1 * reg_lin.params[1] + reg_lin.params[0]

    plt.plot(X1, y_hat, lw=2.5, c='C8')
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('Admitted', fontsize=20)

    plt.show()

    """
    The result is very bad and not plausible; we have possible values which could go below 0% and above 100%
    """

    # plot with a logistic regression curve
    reg_log = sm.Logit(y, X).fit()
    print(reg_log.summary())
    print('Constant/Bias:', reg_log.params[0])
    print('Coef/Weight:', reg_log.params[1])

    def f(x, b0, b1):
        """
        This function squeezes the regression line between 0 and 1
        :param x:
        :param b0:
        :param b1:
        :return:
        """

        """
        Logistic regression function: 𝑝(X) = 𝑒 (β0+β1X1+...+βkXk) / 1+𝑒 (β0+β1X1+...+βkXk)
        """
        return np.array(np.exp(b0 + b1 * x) / (1 + np.exp(b0 + b1 * x)))

    f_sorted = np.sort(f(X1, reg_log.params[0], reg_log.params[1]))
    x_sorted = np.sort(np.array(X1))

    plt.scatter(X1, y, c='C0', alpha=0.3)
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('Admitted', fontsize=20)
    plt.plot(x_sorted, f_sorted, c='C8')

    plt.show()

    # Maximum likelihood estimation (MLE) - MLE tries to maximize the likelihood function by iterating
    # The computer is going through different values, until it finds a model, for which the log likelihood
    # is the highest

    # Log Likelihood-Null:
    x0 = np.ones(168)   # number of observations of our above example
    reg_log_nulls = sm.Logit(y, x0)
    results_log = reg_log_nulls.fit()
    print(results_log.summary())

    print("""
        Watch the both summaries above mentioned carefully: compare LL and LL-Null. You may want to compare
        the log-likelihood of your model with the LL-null, to see if your model has any explanatory power.
        The Log-Likelihood-Ration Test (LLR with p-value) shows how small the p-value is: so the model is
        significant.
        Pseudo-R^2: here McFadden's R^2 - should be between 0.2 and 0.4
    """)


def binary_predictors():
    """
    In the same way we created dummies for a linear regression, we can use binary predictors in a
    logistic regression.
    :return:
    """
    raw_data = pd.read_csv('data_regressions/2.02.+Binary+predictors.csv')
    print(raw_data.head(5))
    data = raw_data.copy()
    data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No': 0})
    data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
    print(data.head())

    # First regression with only gender as predictor

    y_only_gender = data['Admitted']
    x1_only_gender = data[['Gender']]

    x_only_gender = sm.add_constant(x1_only_gender)
    reg_log_only_gender = sm.Logit(y_only_gender, x_only_gender).fit()
    reg_summary_only_gender = reg_log_only_gender.summary()
    print(reg_summary_only_gender)

    print(f"""
            There are only two params: 
            Const./Bias: {reg_log_only_gender.params[0]},
            Weight of Gender: {reg_log_only_gender.params[1]}
            
            Interpretation of the model:
            log(odds) = {reg_log_only_gender.params[0].round(2)} +  {reg_log_only_gender.params[1].round(2)} * Gender
            
            We take two equations:
            log(odds2) = {reg_log_only_gender.params[0].round(2)} +  {reg_log_only_gender.params[1].round(2)} * Gender2
            -
            log(odds1) = {reg_log_only_gender.params[0].round(2)} +  {reg_log_only_gender.params[1].round(2)} * Gender1
            
            =
            Be aware of the logarithm rule:
            log(odds2/odds1) = {reg_log_only_gender.params[1].round(2)} * (Gender2 - Gender1)
            
            Gender 1 is now "0" (so 1 -0 is 1) and log(odds-female/odds-male)
            
            log(odds-female) = {reg_log_only_gender.params[1].round(2)} * log(odds-male)   | exp.
            odds-female = {np.exp(reg_log_only_gender.params[1]).round(2)} * odds-male
            
            The odds to get admitted for a female are 8 times higher than for a male student when we only
            have the gender as a predictor.
            
            That's the interpretation of binary predictor coefficients.
            
            
        """)

    # Second regression: now we add the 'SAT' since we know how good it works as a predictor

    y = data['Admitted']
    x1 = data[['SAT', 'Gender']]

    x = sm.add_constant(x1)
    reg_log = sm.Logit(y, x).fit()
    print(reg_log.summary())

    print(f"""
        There are three params: 
        Const./Bias: {reg_log.params[0]},
        Weight of SAT: {reg_log.params[1]},
        Weight of Gender: {reg_log.params[2]}
        
        Interpretation of the model:
        There was a strong relationship between SAT and admittance:
            log(odds) = {reg_log.params[0].round(2)} + {reg_log.params[1].round(2)} + {reg_log.params[2].round(2)} * Gender
            
            We take two equations:
            log(odds2) = {reg_log.params[0].round(2)} + {reg_log.params[1].round(2)} + {reg_log.params[2].round(2)} * Gender2
            -
            log(odds1) = {reg_log.params[0].round(2)} + {reg_log.params[1].round(2)} + {reg_log.params[2].round(2)} * Gender1
            
            =
            Be aware of the logarithm rule:
            log(odds2/odds1) = {reg_log.params[2].round(2)} * (Gender2 - Gender1)
            
            Gender 1 is now "0" (so 1 -0 is 1) and log(odds-female/odds-male)
            
            log(odds-female) = {reg_log.params[2].round(2)} * log(odds-male)   | exp.
            odds-female = {np.exp(reg_log.params[2]).round(2)} * odds-male
            
            NOW the odds to get admitted for a female are 7 times higher than for a male student (compared
            to the prior 8 times) when we have SAT as a strong predictor. You see, that gender now loses
            explanatory power.
            
            We see that our Log-Likelihood is much higher than the model with only gender, so that the model
            with the SAT is much more significant.
            SAT is an outstanding predictor.
            
            Our gender-p-value is still small (0.022) but not any longer that significant like in the above model.
            
            Result: Given the same SAT score, a female has 7 times higher odds to get admitted.
            But be aware where your data comes from: since universities put their focus on equality, the odds
            will be better for women in MINT/STEM-courses and better for men is social studies or communication.
            
            That's the interpretation of binary predictor coefficients.
    """)

    # Calculate the accuracy - Confusion Matrix CM

    """
    We have a model that predicts values and we also have the actual values (data['Admitted'])
    """
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    predictions = reg_log.predict()
    print('Predicted values by the model:')
    print(predictions)
    print('Actual values:')
    print(np.array(data['Admitted']))
    print("""
        We now have to compare both arrays and look how many percent match:
        Let' take it easy and use sm..pred_table():
        
        It returns a table which compares predicted and actual values
    """)

    confusion_matrix_variable = reg_log.pred_table()
    print(confusion_matrix_variable)

    print('Confusion Matrix - how confused our model was:')
    pd.set_option('display.float_format', lambda x: f'{x:.1f}')
    cm = pd.DataFrame(data=[confusion_matrix_variable[0], confusion_matrix_variable[1]], columns=['Predicted 0', 'Predicted 1'],
                      index=['Actual 0', 'Actual 1'])
    print(cm)
    print("""
        Interpretation:
        For 69 observations the model predicted 0 and the true value was 0, for 4 it predicted 0 but the true value was
        1. For 90 observations the model predicted 1 and the true value was 1 but for 5 observations the
        true value was 0. So we have 9 false predictions and 159 correct predictions.
        This means 159/168 or 94,6% accuracy.
    """)

    def accuracy():
        cm_accuracy = np.array(cm)
        accuracy_train = (cm_accuracy[0, 0] + cm_accuracy[1, 1]) / cm_accuracy.sum()
        return accuracy_train

    accuracy = accuracy()
    print(accuracy.round(2))

    # Testing the model and assessing its accuracy

    """
    Testing is done on a dataset the model has never seen before
    """

    raw_data_testing = pd.read_csv('data_regressions/2.03.+Test+dataset.csv')
    data_testing = raw_data_testing.copy()
    data_testing['Gender'] = data_testing['Gender'].map({'Female': 1, 'Male': 0})
    data_testing['Admitted'] = data_testing['Admitted'].map({'Yes': 1, 'No': 0})
    # print(data_testing)

    # We will use our above model to make predictions based on the test data
    # we will compare those with the actual outcome
    # we will calculate the accuracy
    # we will create a confusion matrix on our own

    # Order is crucial - check it with x: print(x)

    test_actual = data_testing['Admitted']
    test_data = data_testing.drop(['Admitted'], axis=1)
    test_data = sm.add_constant(test_data)
    # in case of reordering needed:
    # test_data = test_data[x.columns.values]
    # print(test_data)  # we need this order to feed our model

    """
    Unfortunately sm.LogitResults.pred_table() does not provide testing as a functionality unless sklearn
    So we have to create it on our own - look at the top of the script, there you find the function
    confusion_matrix()
    """

    cm_test = confusion_matrix(test_data, test_actual, reg_log)
    print(cm_test)
    cm_df = pd.DataFrame(data=cm_test[0], columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
    print(cm_df)

    print('Misclassification rate: ' + str((1 + 1) / 19))

    return reg_log


def logistic_regression_credit_eligibility_including_testing():
    """
    We separate manually the dataset: with 90% trainingsdata we run the logistic regression - create the model
    Then we take the remaining 10% testdata and feed these datapoints into the model and calculate
    the confusion matrix.
    :return:
    """

    raw_data_training = pd.read_csv('data_regressions/Bank_data.csv')
    raw_data_testing = pd.read_csv('data_regressions/Bank_data_testing.csv')

    data_training = raw_data_training.copy()
    data_training.drop(['Unnamed: 0'], axis=1, inplace=True)
    data_training['y'] = data_training['y'].map({'yes': 1, 'no': 0})
    # print(data_training)

    y_training = data_training['y']
    x1_training = data_training['duration']

    x_training = sm.add_constant(x1_training)

    log_reg_training = sm.Logit(y_training, x_training).fit()
    log_results_training = log_reg_training.summary()
    print(log_results_training)

    confusion_matrix_training = log_reg_training.pred_table()
    # print(confusion_matrix_training)

    # Plot the regression:
    plt.scatter(x1_training, y_training, color='C0')
    plt.xlabel('Duration', fontsize=20)
    plt.ylabel('Subscription', fontsize=20)
    plt.show()

    cm_training = pd.DataFrame(confusion_matrix_training, columns=['Predicted Yes', 'Predicted No'],
                               index=['Actual Yes', 'Actual No'])

    def accuracy():
        cm_accuracy_training = np.array(cm_training)
        accuracy_train = (cm_accuracy_training[0, 0] + cm_accuracy_training[1, 1]) / cm_accuracy_training.sum()
        return accuracy_train

    accuracy_training = accuracy()
    #print(accuracy_training.round(2))

    print(f"""
            We see in the scatter plot that duration only is not a very powerful predictor.
            The accuracy of our model is quite bad with {accuracy_training.round(2) * 100} %.
            """)

    # Todo: let's expand the model with more predictors:
    """
        We can be omitting many causal factors in our simple logistic model, so we instead switch to a multivariate
        logistic regression model. Add the ‘interest_rate’, ‘march’, ‘credit’ and ‘previous’ estimators to our model
        and run the regression again.
    

    def accuracy():
        cm_accuracy = np.array(cm)
        accuracy_train = (cm_accuracy[0, 0] + cm_accuracy[1, 1]) / cm_accuracy.sum()
        return accuracy_train

    accuracy = accuracy()
    print(accuracy.round(2))
    """

    estimators = ['interest_rate', 'credit', 'march', 'previous', 'duration']
    X1_all_training = data_training[estimators]
    y_training = data_training['y']

    X_all_training = sm.add_constant(X1_all_training)
    log_reg_training_expanded = sm.Logit(y_training, X_all_training).fit()
    print(log_reg_training_expanded.summary())

    confusion_matrix_training_expanded = log_reg_training_expanded.pred_table()

    cm_training_expanded = pd.DataFrame(confusion_matrix_training_expanded, columns=['Predicted Yes', 'Predicted No'],
                                        index=['Actual Yes', 'Actual No'])

    print(cm_training_expanded)

    # Accuracy and misclassification:

    def accuracy():
        cm_accuracy_training_expanded = np.array(cm_training_expanded)
        accuracy_train_expanded = (cm_accuracy_training_expanded[0, 0] + cm_accuracy_training_expanded[1, 1]) / \
                                   cm_accuracy_training_expanded.sum()
        return accuracy_train_expanded

    accuracy_training_expanded = accuracy()
    #print(accuracy_training_expanded.round(2))

    print(f"""
            The accuracy of our model improved up to {accuracy_training_expanded.round(2) * 100} %.
            """)

    # Confusion matrix with convenient function:
    cm_training_convenient = confusion_matrix(X_all_training, y_training, log_reg_training_expanded)
    print("""
        We see the confusion matrix created by our convenient function:
    """)
    print(cm_training_convenient)

    # Testing the model
    raw_data_testing.drop(['Unnamed: 0'], axis=1, inplace=True)
    raw_data_testing['y'] = raw_data_testing['y'].map({'yes': 1, 'no': 0})
    #print(raw_data_testing)
    #print(raw_data_testing.describe())

    y_test = raw_data_testing['y']
    # We already declared a list called estimators that holds all relevant estimators for our model.
    X1_test = raw_data_testing[estimators]
    X_test = sm.add_constant(X1_test)

    cm_testing_convenient = confusion_matrix(X_test, y_test, log_reg_training_expanded)

    print(f"""
            We see the confusion matrix created by our convenient function for training and testing:
            {cm_training_convenient}
            {cm_testing_convenient}
            
            Looking at the test accuracy we see a number which is a tiny but lower 86,04%, compared
            to the 86,3% for training accuracy.
            
            In general, we always expect the test accuracy to be lower than the training accuracy.
            If the test accuracy is higher, this is just due to luck.
            
        """)


if __name__ == '__main__':
    logistic_regression_basics()
    binary_predictors()
    reg_log = binary_predictors()
    #confusion_matrix(reg_log)
    logistic_regression_credit_eligibility_including_testing()



