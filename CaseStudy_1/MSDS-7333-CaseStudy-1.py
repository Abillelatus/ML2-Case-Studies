#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:13:47 2023
Last Version: Jan 14

@author: Luke Stodgel, Ryan Herrin
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV  # KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error


# Description
program_desc = """
Script that invistigates L1 and L2 Linear Regression Models to predict
the Critical Temperature for superconductivy for different elements.

If this script is downloaded from the github page then ran from the default
location then the files needed are already prepared. If running this script
solo then you will need to define the paths to where train.csv and the
unique_m.csv are using the command line arguments -t and -u.
"""

# Default locations of csv files if not specified in the args
unique_deflt_loc = '../Datasets/superconduct/unique_m.csv'
train_deflt_loc = '../Datasets/superconduct/train.csv'


def log(string_x):
    '''Simple logger for command line formatting'''
    if args.verbose:
        print("[Case Study 1] > {}".format(str(string_x)))
    else:
        pass


def warn(*args, **kwargs):
    '''Supress the sklearn warnings'''
    pass


warnings.warn = warn


def read_in_date(unique_path, train_path):
    '''Read in data from the unique_csv and train_csv and return them as Pandas
    dataFrames.'''
    # Attempt to read in the data as pandas dataframes
    log("Reading in data...")
    try:
        u_dataframe = pd.read_csv(unique_path)
        t_dataframe = pd.read_csv(train_path)
    # If there is a problem reading any of the files throw an error
    except Exception as err:
        print(str(err))

    return(u_dataframe, t_dataframe)


def join_unique_and_train(u_dataframe, t_dataframe):
    ''''Join the unique and train dataframes together.'''
    # Drop the "critical temp" and "material" column from the unique dataframe
    # since it already exists in the train df. We want to avoid accidently
    # joining on these to not join on critical temps.
    log("Removing critical_temp and material columns from unique dataframe...")
    u_dataframe = u_dataframe.drop(['critical_temp', 'material'], axis=1)

    log("Creating combined DataFrame from Unique and Train DataFrames...")
    combined_df = t_dataframe.join(u_dataframe)

    return(combined_df)


def run_preprocessing(dataframe):
    '''View the raw data and make our attempts to normalize and scale the data
    if needed.'''
    # View scatter plot of data to see if there are any outliers
    df = dataframe.copy(deep=True)

    # Remove the critical_temp (out target) column before it gets scaled too
    tmp_target = df['critical_temp']
    df = df.drop(['critical_temp'], axis=1)

    # Get column names to add back
    col_names = df.columns

    # Apply the sklearn stadard scaler
    df_scaler = StandardScaler()
    df = df_scaler.fit_transform(df)

    # Turn it back into a dataframe
    df = pd.DataFrame(data=df, columns=col_names)

    # Add back the target column
    df = df.join(tmp_target)

    return(df)


def get_lasso_alpha(w_dataframe):
    # Create train/test split
    # Split the data into X and y
    X_scaled = w_dataframe.drop('critical_temp', axis=1)
    y = w_dataframe['critical_temp']

    model = Lasso(max_iter=1000)

    # Create standard range
    std_range = np.arange(.01, 11.0, 0.1)

    param_dist = {"alpha": std_range}

    log("Creating best alpha value for L1 (Lasso)")
    # neg_mean_absolute_error suggested 0.01
    grid_search = GridSearchCV(model, param_grid=param_dist, scoring='r2', cv=5)
    grid_search.fit(X_scaled, y)

    log("Best Alpha value for L1 is: {}".format(
        grid_search.best_params_['alpha']))

    # Plot the coefficients
    vi = []
    for i in std_range:
        model.alpha = i
        model.fit(X_scaled, y)
        vi.append(model.coef_)

    V = pd.DataFrame(np.array(vi), columns=X_scaled.columns)
    for i in V.columns:
        plt.plot(std_range, V[i], label=i)

    plt.plot(std_range, vi)
    plt.xlabel('Iteration')
    plt.ylabel('Coefficients')
    plt.title('Convergence of Lasso Coefficients')
    plt.show()

    return(model, grid_search.best_params_)


def get_ridge_alpha(w_dataframe):
    # Create train/test split
    # Split the data into X and y
    X_scaled = w_dataframe.drop('critical_temp', axis=1)
    y = w_dataframe['critical_temp']

    std_dist = np.arange(1, 2000, 10)

    model = Ridge()
    param_dist = {'alpha': std_dist}

    log("Creating best alpha value for L2 (Ridge)")
    grid_search = GridSearchCV(
        model, param_grid=param_dist, scoring='neg_mean_squared_error', cv=5)

    grid_search.fit(X_scaled, y)

    log("Best Alpha value for L2 is: {}".format(
        grid_search.best_params_['alpha']))

    # Plot the coefficients
    vi = []
    for i in std_dist:
        model.alpha = i
        model.fit(X_scaled, y)
        vi.append(model.coef_)

    V = pd.DataFrame(np.array(vi), columns=X_scaled.columns)
    for i in V.columns:
        plt.plot(std_dist, V[i], label=i)

    plt.plot(std_dist, vi)
    plt.xlabel('Iteration')
    plt.ylabel('Coefficients')
    plt.title('Convergence of Ridge Coefficients')
    plt.show()

    return(model, grid_search.best_params_)


def _get_non_zero_coef_list(df, model):
    '''Create a list of coefficients from a model and return it'''
    coef_list = []
    for col in range(len(df.columns[:-1])):
        if model.coef_[col] != 0:
            coef_list.append((df.columns[col], abs(model.coef_[col])))

    # Sort the list from high to low
    coef_list = sorted(coef_list, key=lambda x: x[1], reverse=True)

    return(coef_list)


def get_lasso_mse(dataframe, best_alpha):
    '''Get the mse of the lasso model with multiple runs'''
    # Split the data into train and test sets
    X = dataframe.drop('critical_temp', axis=1)
    y = dataframe['critical_temp']

    # Container to hold the means
    lasso_means = []

    log("Calculating MSE for Lasso...")
    for mse_run in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Instantiate a Lasso regressor
        l1_model = Lasso(alpha=best_alpha['alpha'])

        # Fit the regressor to the data
        l1_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = l1_model.predict(X_test)

        # Compute the MSE of the Lasso model
        lasso_means.append(round(mean_squared_error(y_test, y_pred), 1))

    plt.hist(lasso_means)
    plt.title(
        'Range of Scores of L1 with Lambda = {}'.format(best_alpha['alpha']))
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Count of Runs')
    plt.show()

    return(lasso_means)


def get_ridge_mse(dataframe, best_alpha):
    '''Get the mse of the ridge model with multiple runs'''
    # Split the data into train and test sets
    X = dataframe.drop('critical_temp', axis=1)
    y = dataframe['critical_temp']

    ridge_means = []

    log("Calculating MSE for Ridge")
    for mse_run in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Instantiate a Lasso regressor
        l2_model = Ridge(alpha=best_alpha['alpha'])

        # Fit the regressor to the data
        l2_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = l2_model.predict(X_test)

        # Compute the MSE of the Lasso model
        ridge_means.append(round(mean_squared_error(y_test, y_pred), 1))

    plt.hist(ridge_means)
    plt.title(
        'Range of Scores of L2 with Lambda = {}'.format(best_alpha['alpha']))
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Count of Runs')
    plt.show()

    return(ridge_means)


def display_stats(model_name, df, model, best_alpha, mse_means):
    '''Display the results from each model'''
    # Get the mean for the MSE
    mean_mse = round(np.mean(mse_means), 2)

    # Get a list of the non-zero coefficents from the model
    non_zero_coef = _get_non_zero_coef_list(df, model)

    # Calculate the confidence intervals
    mse_ci = st.norm.interval(
        alpha=0.95, loc=np.mean(mse_means), scale=st.sem(mse_means))
    conf_int_low = round(mse_ci[0], 2)
    conf_int_high = round(mse_ci[1], 2)

    print("\n")
    print('=' * 25)
    print("Results for: {}".format(model_name))
    print("Best Alpha Value: {}".format(str(round(best_alpha['alpha'], 2))))
    print("100 Run MSE: {}".format(str(mean_mse)))
    print("95% C.I. of {} to {}".format(conf_int_low, conf_int_high))
    print("Top features:")
    print("---------------")
    # Print out top 5 variables
    feat_cnt = 0
    for i in range(len(non_zero_coef)):
        if feat_cnt >= 5:
            break
        print('{}: {}'.format(non_zero_coef[i][0], non_zero_coef[i][1]))
        feat_cnt += 1
    print('=' * 25)


if __name__ == "__main__":
    # Grab arguments if provided.
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-u", "--unique_csv", help="Path to the train.csv", metavar='\b')
    parser.add_argument(
        "-t", "--train_csv", help="Path to the unique_m.csv", metavar='\b')
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose", action="store_true")
    args = parser.parse_args()

    # If arguments are provided then treat them as the defualt location for the
    # input files. Other-wise we will check if the files are in the default
    # location. Raise an exception if no files are found.
    if args.unique_csv is not None and args.train_csv is not None:
        # Validate paths
        if os.path.exists(args.unique_csv) and os.path.exists(args.train_csv):
            unique_loc = args.unique_csv
            train_loc = args.train_csv
        else:
            raise Exception("One or more input files could not be found...")

    # Fallback
    elif args.unique_csv is None and args.train_csv is None:
        if os.path.exists(unique_deflt_loc) and os.path.exists(train_deflt_loc):
            unique_loc = unique_deflt_loc
            train_loc = train_deflt_loc

    # Can't find input files. Raise exception
    else:
        input_err = ("One or more input files were not found. Please specify"
                     " unique and train csv files using the -u and -t flags...")

        raise Exception(input_err)

    # Start the main script
    # Read in data
    unique_df, train_df = read_in_date(unique_loc, train_loc)

    # Create a dataframe based on joining the two dataframes
    working_df = join_unique_and_train(unique_df, train_df)

    # Run preprocessing on the data that may include normalization and scaling
    working_df = run_preprocessing(working_df)

    # Create a Lasso model (L1) and return the model and the best alpha
    lasso_model, lasso_alpha = get_lasso_alpha(working_df)

    # Get the MSE from the lasso model
    lasso_mse_means = get_lasso_mse(working_df, lasso_alpha)

    # Create a Ridge model (L2) and return the model and best alpha
    ridge_model, ridge_alpha = get_ridge_alpha(working_df)

    # Generate the MSE from the Ridge Regression
    ridge_mse_means = get_ridge_mse(working_df, ridge_alpha)

    # Display the results
    display_stats(
        "L1 (Lasso)", working_df, lasso_model, lasso_alpha, lasso_mse_means)
    display_stats(
        "L2 (Ridge)", working_df, ridge_model, ridge_alpha, ridge_mse_means)
