# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project is a machine learning project that focuses on predicting customer churn (the loss of customers) for a telecommunications company. The project provides a real-world example of how machine learning can be used to solve business problems and make data-driven decisions.

The project involves working with a dataset of customer information, including demographic information, service usage, and whether or not the customer has churned. The goal is to build a machine learning model that can accurately predict which customers are likely to churn, and then use that model to inform business decisions and strategies to reduce customer churn.

The project teaches the use of Python and popular data science libraries like Pandas, NumPy, and Scikit-learn to preprocess and clean the data, perform exploratory data analysis, and build and evaluate machine learning models. Throughout the project, emphasis is placed on writing clean, modular, and well-documented code, using best practices and following the PEP8 style guide.

## Files and data description

The proyect implements a data analisys  to predict customer churn. The steps are as follows:

1. Loading data
2. Perform exploratory data analysis
3. Encode categorical variables
4. Perform feature engineering
5. Train Random Forest and Logisti Regression models
6. Compare them and find feature importances

All the functions implementd have their own tests and logs, and store the results in their respective folders.

To run the test use the following command:

```
python churn_script_logging_and_tests.py
```

## Running Files

1. Clone the repo in you local machine

```
git clone path
```

2. Create virtual enviroment and install all the requirements

```
mkdir .venv
pipenv install
pipenv install requirements.txt
```

3. Launch churn_library.py script to perform the data analysis

```
python churn_library.py
```
