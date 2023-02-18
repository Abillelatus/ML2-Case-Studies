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
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def create_data(path, create_csv=False):
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

    # Write missing data values to log file
    try:
        with open(os.getcwd() + '/na_log.txt', 'w') as outfile:
            outfile.write(log_str)
            outfile.close()

    except Exception as err:
        print('[Error] Could not create na_log files')
        print(str(err))

    # Remove rows where the taget is na, because we need to train with
    # supervised data
    df = df[df['target'] != np.nan]

    # Impute missing data
    imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
    imputed_data = imputer.fit_transform(df)
    ret_df = pd.DataFrame(imputed_data, columns=headers)

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


if __name__ == "__main__":
    # dir_path = '../Datasets/CS4_data'
    # bnk_data = create_data(dir_path, create_csv=True)
    combined_csv_path = './combined.csv'
    bnk_data = load_csv(combined_csv_path)














