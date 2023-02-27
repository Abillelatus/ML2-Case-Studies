#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case Study 4

Using random forest and xgboost models to predict if a company is going to
go bankrupt.

Created on Fri Feb 17 15:15:40 2023

@author: Ryan Herrin, Luke Stodgel
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.metrics import f1_score
# from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
from sklearn import metrics as mt


program_desc = """
CaseStudy 4: Script that takes in financial data provided from a bank and uses
a Random Forest model and an XGBoost model to determine whether the company
will go bankrupt or not.
"""


def create_data(path, create_csv=False, create_na_log=True):
    """Read in the financial data and return a pandas Dataframe object of the
    data. Param "path" should be the root directory that holds the individual
    files. "create_csv" will export a csv of the combined data if set to True.
    """
    # Create list of files in the dir. Should only be 1-5year data
    raw_files = []
    for root, dirs, files in os.walk(path):
        for name in files:
            raw_files.append(os.path.join(root, name))

    # Create columns names. There are 64 attributes and 1 target column
    headers = []
    for col_name in range(64):
        headers.append('X' + str(col_name + 1))
    headers.append('target')  # Add the target to the end of the headers list

    # List to hold all data
    master_data_list = []

    # Read in the data from the files
    for file in raw_files:
        print('Loading {}'.format(file))
        with open(file, 'r') as infile:
            file_data = infile.read().splitlines()
            # Find the index where the data starts
            start_index = file_data.index('@data') + 1
            master_data_list.extend(file_data[start_index:])

    # All indexes are strings with commas. Need to seperated the values
    for indx in range(len(master_data_list)):
        master_data_list[indx] = master_data_list[indx].split(',')
        # Change from string to float
        for val in range(len(master_data_list[indx])):
            # The first if and elif handles the missing data if present
            # Transforms missing data into np.nan
            if master_data_list[indx][val] == '?':
                master_data_list[indx][val] = np.nan

            elif master_data_list[indx][val] == '':
                master_data_list[indx][val] = np.nan

            else:
                master_data_list[indx][val] = float(
                    master_data_list[indx][val])

    # Create dataframe
    df = pd.DataFrame(master_data_list, columns=headers)

    # Create log file of all missing data
    log_str = 'NA values\n---------\n'
    for col in df.columns:
        # Get number of na values per column
        num_na = df[col].isnull().sum()
        log_str = log_str + '{}: {}\n'.format(col, num_na)

    # Remove rows where the taget is na, because we need to train with
    # supervised data
    df = df[df['target'] != np.nan]

    # Impute missing data
    print("Imputing Missing Data Using KNNImputer...")
    imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
    imputed_data = imputer.fit_transform(df)
    ret_df = pd.DataFrame(imputed_data, columns=headers)
    print("Imputing Complete...")

    # Post impute log update
    log_str = log_str + ('\n' * 3) + ('---' * 20) + ('\n' * 3)
    log_str = log_str + 'Post Impute NA values\n------------------\n'
    for col in df.columns:
        # Get number of na values per column
        num_na = ret_df[col].isnull().sum()
        log_str = log_str + '{}: {}\n'.format(col, num_na)

    if create_na_log:
        # Write missing data values to log file
        try:
            with open(os.getcwd() + '/na_log.txt', 'w') as outfile:
                outfile.write(log_str)
                outfile.close()

        except Exception as err:
            print('[Error] Could not create na_log files')
            print(str(err))

    # Create an output csv if create_csv is true
    if create_csv:
        csv_name = os.getcwd() + '/combined.csv'
        ret_df.to_csv(csv_name, index=False)

    return ret_df


def load_csv(csv_path):
    """Load in combined csv file if generated previously"""
    try:
        return pd.read_csv(csv_path)

    except Exception as err:
        print(str(err))


def display_model_metrics(metrics):
    '''Print out formatted metrics for the model'''
    # Get the means
    acc_mean = round(np.mean(metrics[0]), 3)
    prec_mean = round(np.mean(metrics[1]), 3)
    rec_mean = round(np.mean(metrics[2]), 3)
    f1_mean = round(np.mean(metrics[3]), 3)
    # Get the variance
    acc_var = round(np.var(metrics[0]), 6)
    prec_var = round(np.var(metrics[1]), 6)
    rec_var = round(np.var(metrics[2]), 6)
    f1_var = round(np.var(metrics[3]), 6)

    print()
    print('Accuracy:  Mean = {} | Variance = {}'.format(acc_mean, acc_var))
    print('Precision: Mean = {} | Variance = {}'.format(prec_mean, prec_var))
    print('Recall:    Mean = {} | Variance = {}'.format(rec_mean, rec_var))
    print('F1_Score:  Mean = {} | Variance = {}'.format(f1_mean, f1_var))
    print()


def get_model_results_and_plots(
        dataframe, model, best_params, thresh, num_tests):
    '''Common function to get results from a passed model'''
    # Split the DataFrame into training and testing sets

    # Remove the 'target' column from the DataFrame and assign the result to X
    X = dataframe.drop('target', axis=1)
    # Assign the 'target' column to y
    y = dataframe['target'].astype(int)

    # Names for the charts
    names = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Container to hold the means
    accuracy_means = []
    precision_means = []
    recall_means = []
    f1_means = []
    # mnb_auc_means = []

    # Container used to feed into charts
    mean_values = []

    # Run the model with final stats multiple times to get a mean of all
    for run in range(num_tests):
        # Create test train split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )

        # Instantiate a model from the params
        test_model = model(**best_params)

        # There scores
        # Fit the Model to the data
        test_model.fit(X_train, y_train)

        # If a threshold was provided then use it
        if thresh is not None:
            # find the best threshold
            threshold = thresh
            # calculate class labels using probabilities
            y_hat = test_model.predict_proba(X_test)[:, 1] >= threshold
            y_hat = np.where(y_hat, 1, 0)

        else:
            y_hat = test_model.predict(X_test)

        # Compute the metrics of the model
        # fpr, tpr, thresholds = mt.roc_curve(y_test, y_hat_proba, pos_label=1)
        accuracy_means.append(round(accuracy_score(y_test, y_hat), 3))
        precision_means.append(round(precision_score(y_test, y_hat), 3))
        recall_means.append(round(recall_score(y_test, y_hat), 3))
        f1_means.append(round(mt.f1_score(y_test, y_hat), 3))

    mean_values.append(np.mean(accuracy_means))
    mean_values.append(np.mean(precision_means))
    mean_values.append(np.mean(recall_means))
    mean_values.append(np.mean(f1_means))

    # Create the barplot figure
    fig, ax = plt.subplots()
    bars = ax.barh(names, mean_values)
    ax.bar_label(bars)
    plt.yticks(names)

    # Seperate titles if using threshold or not
    if thresh is not None:
        plt.title(
            'Average Mean Scores |'
            ' ProbThreshold = {} |'
            ' Num_Runs={}'.format(threshold, num_tests))
    else:
        plt.title(
            'Average Mean Scores |'
            ' Num_Runs={}'.format(num_tests))

    plt.xlabel('Percentage', fontsize=11, color='blue')
    plt.ylabel('Metrics', fontsize=11, color='blue')
    plt.show()

    # Create list of lists for the boxplot
    values = [accuracy_means, precision_means, recall_means, f1_means]

    # Create Boxplot of model metric variances
    bx_plt = plt.figure()
    bx_plt.suptitle('Model Metric Variances over {} runs'.format(num_tests))
    ax = bx_plt.add_subplot(111)
    plt.boxplot(values)
    plt.grid()
    plt.tight_layout(pad=1.5)
    ax.set_xticklabels(names)
    plt.xlabel('Model Metric',
               fontsize=11, color='blue')
    plt.ylabel('Score (as a percentage)', fontsize=11, color='blue')
    plt.show()

    return values


if __name__ == "__main__":
    # Grab arguments
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-d", "--directory", help="Path to CS4_Directory", metavar='\b')
    args = parser.parse_args()

    dir_path = ''

    # Validate the directory path exists
    if args.directory is not None:
        # Validate paths
        if os.path.exists(args.directory):
            # validate blank argument that might default to the home dir
            if len(args.directory) > 1:
                dir_path = args.directory
            else:
                raise Exception("-d argument cannot be blank...")
        else:
            raise Exception("Path Provided does not exists or is invalid...")

    bnk_data = create_data(dir_path, create_csv=False, create_na_log=False)

    # For loading in a csv file that is already created from the financial data
    '''
    combined_csv_path = './combined.csv'
    bnk_data = load_csv(combined_csv_path)
    '''

    best_params_rf = {'n_estimators': 100, 'min_samples_split': 10,
                      'min_samples_leaf': 1, 'max_features': None,
                      'max_depth': 15, 'criterion': 'entropy',
                      'class_weight': None}

    threshold_avg_rf = 0.298

    best_params_xgb = {'subsample': 0.8, 'scale_pos_weight': 10,
                       'reg_lambda': 1.0, 'reg_alpha': 1.0,
                       'objective': 'binary:logistic', 'n_estimators': 500,
                       'max_depth': 7, 'learning_rate': 0.05, 'gamma': 1,
                       'eval_metric': 'error', 'eta': 0.5,
                       'colsample_bytree': 0.5}

    threshold_avg_xgb = 0.564

    # RF model using traditional predict() instead of predict_proba()
    print("Creating RandomForest...")
    model_metrics_rf_trad = get_model_results_and_plots(
        bnk_data, RandomForestClassifier, best_params_rf, None, 1
    )

    # RF model using predict_proba() instead of predict()
    print("Creating RandomForest with Threshold...")
    model_metrics_rf = get_model_results_and_plots(
        bnk_data, RandomForestClassifier, best_params_rf, threshold_avg_rf, 1
    )

    # XGB model using traditional predict() instead of predict_proba()
    print("Creating XGBoost...")
    model_metrics_xgb_trad = get_model_results_and_plots(
        bnk_data, xgb.XGBClassifier, best_params_xgb, None, 1
    )

    # XGB model using predict_proba() instead of predict()
    print("Creating XGBoost using Threshold...")
    model_metrics_xgb = get_model_results_and_plots(
        bnk_data, xgb.XGBClassifier, best_params_xgb, threshold_avg_xgb, 1
    )

    # Display the results on Console
    print("Random Forest - No Threshold")
    display_model_metrics(model_metrics_rf_trad)

    print("Random Forest - With Threshold")
    display_model_metrics(model_metrics_rf)

    print("XGBoost - No Threshold")
    display_model_metrics(model_metrics_xgb_trad)

    print("XGBoost - With Threshold")
    display_model_metrics(model_metrics_xgb)

    print("Script Complete...")
