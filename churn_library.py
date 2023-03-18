'''
Implementation of Churn Customer Analysis and definition of helper funcitons for the purpose

Author: AlexDLP

Date  : March 2023
'''


# import libraries
import warnings
import os
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

warnings.filterwarnings("ignore")


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    # read data
    dataframe = pd.read_csv(pth)

    # remove unnamed columns if so
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]

    # check there are no nulls
    dataframe.isnull().sum().sum()

    # calculate churn
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # remove non-used variables
    dataframe = dataframe.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)

    # show dataframe
    print(dataframe.head())

    return dataframe


def perform_eda(dframe: pd.DataFrame, output_pth: str):
    '''
    perform eda on df and save figures to images folder
    input:
            dframe: pandas dataframe
            output_pth: path to store the figure

    output:
            None
    '''

    # print the shape of the dataframe
    print(f"The shape of the dataframe is {dframe.shape}")

    # describe de dataset
    print(dframe.describe())

    ### --- SAVE PLOTS ---###
    # UNIVARIATE QUANTITATIVE

    # custome churn plot
    plt.figure(figsize=(6, 3))
    dframe['Churn'].hist()
    plt.ylabel('Churn level')
    plt.title('Customer Churn')
    plt.tight_layout()
    plt.savefig(output_pth + 'Customer_Churn.jpg')

    # customer age distribution plot
    plt.figure(figsize=(6, 3))
    dframe['Customer_Age'].hist()
    plt.ylabel('Customer_Age')
    plt.title('Customer Age Distribution')
    plt.savefig(output_pth + 'Customer_Age_Distribution.jpg')

    # cost density plot
    plt.figure(figsize=(6, 3))
    sns.histplot(dframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(output_pth + 'Total_Transaction_Density.jpg')

    # UNIVARIATE CATEGORICAL
    # Marital status plot
    plt.figure(figsize=(6, 3))
    dframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.ylabel('Marital_Status')
    plt.title('Marital Status Distribution')
    plt.savefig(output_pth + 'Marital_Status_Distribution.jpg')

    # BIVARIATE PLOT
    # heatmat
    plt.figure(figsize=(6, 3))
    sns.heatmap(dframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(output_pth + 'Heatmap.jpg')

    # bivariate correlation
    plt.figure(figsize=(6, 3))
    sns.scatterplot(
        y='Total_Ct_Chng_Q4_Q1',
        x='Total_Amt_Chng_Q4_Q1',
        data=dframe)
    plt.title('High_positive_correlation_bivariate_plot')
    plt.savefig(output_pth + '/High_positive_correlation_bivariate_plot.jpg')


def cat_quant_colums(data: pd.DataFrame):
    """Separates the columns of a dataframe into categorical or numerical variables.
    Returns a list of each of them.

    Args:
        data (DataFrame): pandas Dataframe

    returns:
        cat_columns (list): a list with the names of the categorical variables of the dataframe
        quant_columns (list): a list with the names of the numerical variables of the dataframe
    """

    cat_columns = []
    quant_columns = []

    for colum_name in data.columns:
        if data[colum_name].dtype == 'O':
            cat_columns.append(colum_name)

        else:
            quant_columns.append(colum_name)

    return cat_columns, quant_columns


def encoder_helper(dataframe: pd.DataFrame, category_lst: list, response: str):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
            variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    # copy dataframe
    df_copy = dataframe.copy()

    # loop to encode variables
    for var_name in category_lst:

        empty_list = []
        var_name_groups = df_copy.groupby(
            var_name).mean(numeric_only=True)['Churn']

        for value in df_copy[var_name]:
            empty_list.append(var_name_groups.loc[value])

        new_variable = var_name + '_' + response

        df_copy[new_variable] = empty_list

    return df_copy


def perform_feature_engineering(dataframe: pd.DataFrame, response: str):
    '''
    input:
            df: pandas dataframe
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''

    # obtan categorical and numerical variable lists
    categorical_columns, _ = cat_quant_colums(dataframe)

    # encode categirical variables
    new_df = encoder_helper(dataframe, categorical_columns, response)

    # select response and dependent variables
    target_var = new_df['Churn']  # target
    dependen_vars = new_df.drop(['Churn'], axis=1)  # remove response variable
    dependen_vars = dependen_vars.drop(
        categorical_columns, axis=1)  # remove categorical variables

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        dependen_vars, target_var, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(
        y_train,
        y_test,
        y_train_preds,
        y_test_preds,
        model_identifier):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training true values
            y_test:  test true values
            y_train_preds: training predictions from logistic regression
            y_train_preds: training predictions from random forest
            y_test_preds: test predictions from logistic regression
            model_identifier: model's name

    output:
            None
    '''

    # scores
    print(model_identifier + 'results')
    print('test results')
    print(classification_report(y_test, y_test_preds))
    print('train results')
    print(classification_report(y_train, y_train_preds))

    plt.rc('figure', figsize=(5, 5))
    plt.figure(figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str(model_identifier + ' Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    output_pth = r"./images/results/"
    plt.savefig(output_pth + model_identifier + '.jpg')


def feature_importance_plot(model, X_train, X_test, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''

    # Random forest tree explainer
    plt.figure(figsize=(15, 5))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar",show=False)
    plt.savefig(output_pth + 'Explainable_variables.jpg')

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_train.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_train.shape[1]), names, rotation=45)
    plt.savefig(output_pth + 'Feature_importance_plot.jpg')


def train_models(X_train, X_test, y_train, y_test, result_output_path):
    '''
    train, store model results: images + scores, and store models
    input:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    output:
            None
    '''

    # grid search
    rf_classifier = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # grid search on with random forest
    rf_grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=param_grid,
        cv=5)
    rf_grid_search.fit(X_train, y_train)

    # logistic classifier
    lr_classifier.fit(X_train, y_train)

    # save best models
    joblib.dump(
        rf_grid_search.best_estimator_,
        result_output_path["MODELS"] +
        'Random_forest.pkl')
    joblib.dump(
        lr_classifier,
        result_output_path["MODELS"] +
        'logistic_model.pkl')

    # load the models again
    rf_classifier = joblib.load(
        result_output_path["MODELS"] +
        'Random_forest.pkl')
    lr_classifier = joblib.load(
        result_output_path["MODELS"] +
        'logistic_model.pkl')

    # calculate predictions
    y_train_preds_rf = rf_classifier.predict(X_train)
    y_test_preds_rf = rf_classifier.predict(X_test)

    y_train_preds_lr = lr_classifier.predict(X_train)
    y_test_preds_lr = lr_classifier.predict(X_test)

    # classification report logistic regression
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_test_preds_lr,
                                'Logistic Regression')

    # classification report random forest
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_rf,
                                y_test_preds_rf,
                                'Random Forest')

    # roc curve plot
    plt.figure(figsize=(8, 4))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(
        rf_classifier,
        X_test,
        y_test,
        ax=axis,
        alpha=0.8)
    RocCurveDisplay.from_estimator(
        lr_classifier, X_test, y_test, ax=axis, alpha=0.8)
    plt.title("Roc Curve Results")
    plt.savefig(result_output_path["MODEL_RESULTS"] + 'Roc_curve_results.jpg')

    # feature importance plots
    feature_importance_plot(
        rf_classifier,
        X_train,
        X_test,
        result_output_path["MODEL_RESULTS"])


if __name__ == "__main__":

    # PATHS for file loading and result saving
    BANK_DATA = r"./data/bank_data.csv"  # path file to load
    RESULTS_PATH = {"MODELS": r"./models/",
                    "EDA_RESULTS": r"./images/eda/",
                    "MODEL_RESULTS": r"./images/results/"}

    # load dataframe
    print("\nLoading data...")
    DATA = import_data(BANK_DATA)

    # perform EDA
    print("\nPerforming EDA...")
    perform_eda(DATA, RESULTS_PATH["EDA_RESULTS"])

    # FEATURE ENGINEERING
    print("\nPerforming FEATURE ENGINEERING...")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        DATA, "Churn")

    # TRAIN MODELS
    print("\nTRAINING MODELS...")
    train_models(X_train, X_test, y_train, y_test, RESULTS_PATH)

    print("\n\nProgram finished")
