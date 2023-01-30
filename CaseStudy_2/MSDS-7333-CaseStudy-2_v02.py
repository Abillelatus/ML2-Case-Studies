#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:57:00 2023

@author: Luke Stodgel, Ryan Herrin
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
from sklearn import metrics as mt
from sklearn.metrics import confusion_matrix, f1_score


# Description
program_desc = """
Case Study to build a classifier using logistic regression to predict
hospital readmittance.

Specify location of the diabetic_data.csv using the -f <file path> flag.
"""

# Default location for the diabeic_data.csv
diabts_dflt_loc = '../Datasets/dataset_diabetes/diabetic_data.csv'


def read_in_data(csv_path):
    '''Read in the diabetes data csv. Takes filepath as a string.
    returns : pandas DataFrame'''
    try:
        log('Reading in Data...')
        df = pd.read_csv(csv_path)

    except Exception as err:
        print(str(err))

    return df


def warn(*args, **kwargs):
    '''Supress the sklearn warnings'''
    pass


# Warning supression
warnings.warn = warn


def log(x):
    '''Logger to format output'''
    print('[CaseStudy_2] > {}'.format(str(x)))


def perform_transform(dataframe, categorical_var_list):
    '''Performs trasnformation on the data.
    returns : pandas DataFrame'''
    log('Preforming Data Transformation...')
    # Create a working copy of the passed dataframe
    wrk_df = dataframe.copy()
    catg_lst = categorical_var_list

    # Replace question marks with the np.nan values
    wrk_df.replace('?', np.nan, inplace=True)

    # Columns to drop
    col_to_drop = [
        'weight',
        'encounter_id',
        'patient_nbr',
        'examide',
        'citoglipton',
        'payer_code',
        'medical_specialty']

    # drop multiple columns by name
    wrk_df = wrk_df.drop(columns=col_to_drop)

    # Verify that the dropped columns are not in the list for categorical
    for col in col_to_drop:
        if col in catg_lst:
            catg_lst.remove(col)
        else:
            pass

    # Calculate and impute the mode for the race feature
    mode_race = wrk_df['race'].mode().values[0]
    wrk_df['race'].fillna(mode_race, inplace=True)

    # We are going to leave diag_1/2/3 as NA because the column is made up
    # of a mix of strings and ints and we can't take the mean of a non-int.
    # replace missing values in the column with the string 'Unknown'
    wrk_df['diag_1'] = wrk_df['diag_1'].fillna('Unknown')
    wrk_df['diag_2'] = wrk_df['diag_2'].fillna('Unknown')
    wrk_df['diag_3'] = wrk_df['diag_3'].fillna('Unknown')

    # Label encoding categorical variables so they will work with logistic
    # regression
    le = LabelEncoder()
    # create a OrdinalEncoder object
    oe = OrdinalEncoder()

    # Create a dataframe for only categorical data
    df_cat = wrk_df[catg_lst]

    # fit and transform the categorical column using the OrdinalEncoder
    df_cat_encoded = oe.fit_transform(df_cat)
    # Transform it back to a DataFrame
    df_cat = pd.DataFrame(df_cat_encoded, columns=catg_lst)

    # Drop the original data from the original working df
    wrk_df = wrk_df.drop(catg_lst, axis=1)

    # Join the encoded columns with the rest of the data
    wrk_df = wrk_df.join(df_cat)

    # Grab the target column
    y_trgt = wrk_df['readmitted']
    # fit and transform the target column using the LabelEncoder
    y_trgt = le.fit_transform(y_trgt)
    y_trgt = pd.DataFrame(y_trgt, columns=['readmitted'])

    # Start standardizing/scaling the continous vars
    # Break off the categorical data to be added back later
    wrk_df_cat = wrk_df[catg_lst]

    # Create a numerical dataframe by dropping the categorical data
    catg_lst_w_target = catg_lst
    catg_lst_w_target.append('readmitted')
    wrk_df_num = wrk_df.drop(catg_lst_w_target, axis=1)

    # Create scaler object and scale numerical data
    scaler = StandardScaler().fit(wrk_df_num)
    wrk_scaled = scaler.transform(wrk_df_num)
    wrk_df_scaled = pd.DataFrame(wrk_scaled, columns=wrk_df_num.columns)

    # Combine the categorical and numerical back together
    wrk_df = wrk_df_scaled.join(wrk_df_cat)

    return wrk_df, y_trgt


def get_lasso_model(data, target):
    '''Feature selection model using Lasso. Returns best model'''
    log("Getting best model using Lasso...")
    # start timer
    start_time = time.perf_counter()

    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2)

    # Create Pipeline to feed into gridsearch
    pipeline = Pipeline(
        [('model', LogisticRegression(solver="liblinear",
                                      penalty="l1",
                                      multi_class='ovr'))])

    # value of the AUC in a 5-folds cross-validation and select the value
    # of C thatminimizes such average performance metrics
    search = GridSearchCV(
        pipeline,
        {'model__C': np.arange(.001, .05, .005)},
        cv=5, scoring="roc_auc_ovr", verbose=1)

    search.fit(x_train, y_train)

    best_score = search.best_score_

    # Get the best coefficients and create a list
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    list_of_important_variables = np.array(
        data.columns)[importance[0] > 0]
    # The list
    importantVariablesDF = data[list_of_important_variables]

    # Save the selected features into LASSO data frame
    final_features = importantVariablesDF.columns
    x_train_lasso = x_train[final_features]
    x_test_lasso = x_test[final_features]

    # Define the classifier
    logreg_lasso = LogisticRegression(multi_class='ovr', C=.04)

    # Fit the classifier to the training data
    logreg_lasso.fit(x_train_lasso, y_train)

    # end timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Predict the class labels for the test data
    y_pred_lasso = logreg_lasso.predict(x_test_lasso)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_lasso)
    # Create a DataFrame to store the confusion matrix
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        columns=["Predicted <30", "Predicted >30", "Predicted NO"],
        index=["Actual <30", "Actual >30", "Actual NO"])

    # calculate metrics
    logreg_acc_lasso = mt.accuracy_score(y_test, y_pred_lasso)

    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred_lasso, average='weighted')

    # Store Results
    lasso_results = [logreg_lasso, elapsed_time, logreg_acc_lasso,
                     f1, conf_matrix_df, 'Lasso Features', best_score]

    # Show the results
    _display_model_info(lasso_results, data)

    return lasso_results


def get_full_model(data, target):
    '''Get a model using the all features from the transformed data'''
    log("Getting best model using full feature set...")
    # Create train and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2)

    # start timer
    start_time = time.perf_counter()

    # Create Pipeline to feed into gridsearch
    pipeline = Pipeline(
        [('model', LogisticRegression(solver="liblinear",
                                      penalty="l1",
                                      multi_class='ovr'))])

    # value of the AUC in a 5-folds cross-validation and select the value
    # of C thatminimizes such average performance metrics
    search = GridSearchCV(
        pipeline,
        {'model__C': np.arange(.001, .05, .005)},
        cv=5, scoring="roc_auc_ovr", verbose=1)

    search.fit(x_train, y_train)

    best_score = search.best_score_

    # Define the classifier
    logreg = LogisticRegression(multi_class='ovr',
                                C=0.04)

    # Fit the classifier to the training data
    logreg.fit(x_train, y_train)

    # end timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Predict the class labels for the test data
    y_pred_full = logreg.predict(x_test)

    # calculate metrics
    logreg_acc = mt.accuracy_score(y_test, y_pred_full)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_full)

    # Create a DataFrame to store the confusion matrix
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        columns=["Predicted <30", "Predicted >30", "Predicted NO"],
        index=["Actual <30", "Actual >30", "Actual NO"])

    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred_full, average='weighted')

    full_results = [logreg, elapsed_time, logreg_acc, f1, conf_matrix_df,
                    'Model with Full Features', best_score]

    _display_model_info(full_results, data)

    return(full_results)


def _display_model_info(model_info, data):
    '''Display the model information we want to see'''
    # Input data should be as a list and contain the following:
    # [model, elapsed_time, accuracy, f1_score, confusion_mtrx, config_name]
    model = model_info[0]
    total_time = model_info[1]
    acc = model_info[2]
    f1 = model_info[3]
    conf_mtrx = model_info[4]
    config_name = model_info[5]
    best_score = model_info[6]
    top_five_coef = ''

    # Get Top coefficients
    col_names = data.columns
    coefs = model.coef_[0]
    # Get the top 5
    top_5 = np.argsort(coefs)[::-1][:5]
    coef_dict = dict(zip(col_names, coefs))
    for i in top_5:
        top_five_coef = top_five_coef + '{} : {}\n'.format(
            col_names[i], coef_dict[col_names[i]])

    print('*' * 80)
    print("Configuration: {}".format(config_name))
    print('Total run time (seconds) : {}'.format(total_time))
    print('Accuracy: {}'.format(acc))
    print('F1 Score: {}'.format(f1))
    print('AUC: {}'.format(best_score))
    print('Confusion Matrix:\n{}'.format(conf_mtrx))
    print('Top Five Features: \n{}'.format(top_five_coef))
    print('*' * 80)


if __name__ == "__main__":
    # Grab arguments
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-f", "--data_file", help="Path to diabetic_data.csv", metavar='\b')
    args = parser.parse_args()

    # If arguments are provided then treat them as the defualt location for
    # input files. Other-wise we will check if the files are in the default
    # location. Raise an exception if no files are found.
    if args.data_file is not None:
        # Validate paths
        if os.path.exists(args.data_file):
            diabts_csv_loc = args.data_file
        else:
            raise Exception("Input files could not be found...")

    # If no arguments are provided, see if file is in default location
    elif args.data_file is None:
        if os.path.exists(diabts_dflt_loc):
            diabts_csv_loc = diabts_dflt_loc

    # Can't find input files. Raise exception
    else:
        input_err = (
            "Error: diabetic_data.csv could not be found. Please specifiy",
            "location using -f ")

        raise Exception(input_err)

    catg_vars = [
        'race', 'gender', 'age', 'weight', 'diag_1', 'diag_2', 'diag_3',
        'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
        'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change', 'diabetesMed']

    # Read in data
    df_orig = read_in_data(diabts_csv_loc)

    # Transform the data and get the target column
    df_modified, y_target = perform_transform(df_orig, catg_vars)

    # Run the lasso model and get the best estimates back
    model_lasso = get_lasso_model(df_modified, y_target)

    # Run the model with full features
    model_full = get_full_model(df_modified, y_target)
