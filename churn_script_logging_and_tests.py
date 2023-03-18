'''
Test and logging functions for churn_library.py

Author: AlexDLP

Date  : March 2023
'''


import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        assert 'Unnamed' not in dataframe.columns
        logging.info("Testing import_data: Dataframe free of Unnamed columns")

    except AssertionError as err:
        logging.error(
            "Testing import_data: Dataframe contains Unnamed columns")

    try:
        assert dataframe.isnull().sum().sum() == 0

        logging.info("Testing import_data: Dataframe free of null values")

    except AssertionError as err:
        logging.error("Testing import_data: Dataframe contains null values")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    dataframe = cls.import_data("./data/bank_data.csv")

    try:
        perform_eda(dataframe, r"./images/eda/")
        logging.info("Testing perform_eda: SUCCESS")

    except KeyError as err:
        logging.error("The function has raisen an error")
        raise err

    # check if customer churn plot exists
    try:
        assert os.path.isfile("./images/eda/Customer_Churn.jpg") is True
        logging.info("Customer_Churn.jpg plot has been created")

    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err

    # check if customer age distribution plot exists
    try:
        assert os.path.isfile(
            "./images/eda/Customer_Age_Distribution.jpg") is True
        logging.info("Customer_Age_Distribution.jpg plot has been created")

    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err

    # check if Total_Transaction_Density plot exists
    try:
        assert os.path.isfile(
            "./images/eda/Total_Transaction_Density.jpg") is True
        logging.info("Total_Transaction_Density.jpg plot has been created")

    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err

    # check if Marital_Status_Distribution plot exists
    try:
        assert os.path.isfile(
            "./images/eda/Marital_Status_Distribution.jpg") is True
        logging.info("Marital_Status_Distribution.jpg plot has been created")

    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err

    # check if Heatmap plot exists
    try:
        assert os.path.isfile("./images/eda/Heatmap.jpg") is True
        logging.info("Heatmap.jpg plot has been created")

    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err

        # check if High_positive_correlation_bivariate_plot plot exists
    try:
        assert os.path.isfile(
            "./images/eda/High_positive_correlation_bivariate_plot.jpg") is True
        logging.info(
            "High_positive_correlation_bivariate_plot.jpg plot has been created")

    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err


def test_cat_quant_colums(dataframe):
    '''
    test cat quant colums helper
    '''

    try:
        cat_columns, numeric_colums = cls.cat_quant_colums(dataframe)
        logging.info("Testing cat_quant_colums: SUCCESS")

    except KeyError as err:
        logging.error("An error happened in cat_quant_columns")
        raise err

    # cherk if all categorical colums are categorical
    try:
        for variable in cat_columns:
            assert dataframe[variable].dtype == 'O'
        logging.info("All categorical variables match with type")
    except TypeError as err:
        logging.error(
            "Not al variables in categorical variables colum are categorical")
        raise err

    # check if all numerical columns are numerical
    try:
        for variable in numeric_colums:
            assert dataframe[variable].dtype in ('float64','int64')
        logging.info("All numerical variables match with type")
    except TypeError as err:
        logging.error(
            "Not al variables in numerical variables colum are numerical")
        raise err


def test_encoder_helper(dataframe):
    '''
    test encoder helper
    '''

    cat_columns, numerical_colums = cls.cat_quant_colums(dataframe)

    try:
        encoded_dataframe = cls.encoder_helper(dataframe, cat_columns, 'Churn')
        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        logging.error("An error happened in cat_quant_columns")
        raise err

    # new variables mustn't be categorical and their name must be different
    try:
        for variable in cat_columns:
            assert encoded_dataframe[variable + '_' + 'Churn'].dtype != 'O'
        logging.info("All categorical variables have been encoded")
    except AssertionError as err:
        logging.error("Some categorical variable havent been encoded")
        raise err

    # the rest of the numerical variables should be there
    try:
        for variable in numerical_colums:
            assert variable in encoded_dataframe.columns
        logging.info(
            "All numerical variables exist in the new encoded dataframe")
    except AssertionError as err:
        logging.error("Some numerical variable missing in the new dataframe")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    # load dataframe
    dataframe = cls.import_data("./data/bank_data.csv")

    try:
        test_cat_quant_colums(dataframe)
        test_encoder_helper(dataframe)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe, 'Churn')

        logging.info("Testing test_perform_feature_engineering: SUCCESS")

    except KeyError as err:
        logging.error("An error happened in perform_feature_engineering")
        raise err

    # check that Churn variable is not in dependent variables and ensure that
    # is the only variable in target
    try:
        assert 'Churn' not in x_train.columns
        assert 'Churn' not in x_test.columns
        assert 'Churn' == y_train.name
        assert 'Churn' == y_test.name
        logging.info(
            "Churn is only the target variable and does not exist in dependent datasets")
    except AssertionError as err:
        logging.error(
            'Churn is not the only target variable or still exists in depedent datasets')
        raise err

    # Assert 30% data split
    try:

        assert x_train.shape[0] + x_test.shape[0] == dataframe.shape[0]
        assert y_train.shape[0] + y_test.shape[0] == dataframe.shape[0]
        assert round(y_test.shape[0] / dataframe.shape[0], 2) == 0.3
        assert round(x_test.shape[0] / dataframe.shape[0], 2) == 0.3
        logging.info("The train test split is done at 30%")
    except AssertionError as err:
        logging.error("The train test split is not done at 30%")
        raise err


def test_train_models(train_models, paths):
    '''
    test train_models
    '''
    try:
        # load dataframe
        dataframe = cls.import_data("./data/bank_data.csv")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            dataframe, "Churn")
        train_models(x_train, x_test, y_train, y_test, paths)

        logging.info("Testing train_models: SUCCESS")

    except KeyError as err:
        logging.error("An error happened in train_models")
        raise err

    # Check if the models are saved in models folder
    try:
        assert os.path.isfile(paths["MODELS"] + 'logistic_model.pkl') is True
        assert os.path.isfile(paths["MODELS"] + 'Random_forest.pkl') is True
        logging.info(
            'Logistec regression and random forest models have been created')
    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err

    # Check if the classification reports are images/results folder
    try:
        assert os.path.isfile(
            paths["MODEL_RESULTS"] +
            'Logistic Regression.jpg') is True
        assert os.path.isfile(
            paths["MODEL_RESULTS"] +
            'Random Forest.jpg') is True
        logging.info(
            'Logistec regression and random forest classification reports have been created')
    except AssertionError as err:
        logging.error('The file is not in folder')
        raise err

    # Check if the Explainable variables curve plot is in images/results folder
    try:
        assert os.path.isfile(
            paths["MODEL_RESULTS"] +
            'Explainable_variables.jpg') is True
        logging.info('Explainable_variables has been created')
    except AssertionError as err:
        logging.error('Explainable_variables file is not in folder')
        raise err

    # Check if the Feature importance plot is in images/results folder
    try:
        assert os.path.isfile(
            paths["MODEL_RESULTS"] +
            'Feature_importance_plot.jpg') is True
        logging.info('Feature_importance_plot has been created')
    except AssertionError as err:
        logging.error('Feature_importance_plot file is not in folder')
        raise err

    # Check if the Roc curve plot is in images/results folder
    try:
        assert os.path.isfile(
            paths["MODEL_RESULTS"] +
            'Roc_curve_results.jpg') is True
        logging.info('Roc_curve_plot has been created')
    except AssertionError as err:
        logging.error('Roc_curve_plot file is not in folder')
        raise err


if __name__ == "__main__":

    RESULTS_PATH = {"MODELS": r"./models/",
                    "EDA_RESULTS": r"./images/eda/",
                    "MODEL_RESULTS": r"./images/results/"}
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models, RESULTS_PATH)
