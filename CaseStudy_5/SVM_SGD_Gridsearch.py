#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:53:51 2023

@author: Ryan Herrin, Luke Stodgel
"""

import os
import csv
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import cross_validate, KFold
from sklearn.compose import ColumnTransformer
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn import metrics as mt
# from sklearn.model_selection import train_test_split
# import seaborn as sns
import pandas as pd
from IPython import get_ipython

# For Gridsearch Testing
from BFM_Search import BFMSearch

# Needed for showing plots inline when using the spyder IDE
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception as err:
    err = err
    pass


program_desc = """
Case Study 5. Using SVM and SVG to analyze internet traffic and determine
what action to perform on the traffic.
"""


def load_log2_data(file_path):
    """Loads the data. Takes file path as a string and returns a dictionary
    of data and target."""
    # Var to be returned
    ret_data = dict()

    # Data from the csv
    raw_data = []
    targets = []

    # Read in data
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            raw_data.append(row)

    # Add the data to the returned dictionary then remove it from the data
    ret_data["headers"] = raw_data[0]
    raw_data.pop(0)

    # Get the index of the target column
    trgt_indx = ret_data["headers"].index("Action")
    # Remove and add the target data to its own list
    for row in raw_data:
        targets.append(row.pop(trgt_indx))

    # Remove the target column from the headers
    ret_data["headers"].pop(trgt_indx)

    # Add data to the returned dictionary
    ret_data["data"] = np.asarray(raw_data)
    ret_data["target_raw"] = targets
    ret_data["target_mapping"] = {
        "allow": 0,
        "deny": 1,
        "drop": 2,
        "reset-both": 3
    }

    # Encode the target data
    encoded_targets = []
    for trgt in targets:
        encoded_targets.append(ret_data["target_mapping"][trgt])

    ret_data["target"] = np.asarray(encoded_targets)

    # convert arrays to smaller values
    ret_data["data"] = ret_data["data"].astype(np.dtype("u2"))
    ret_data["target"] = ret_data["target"].astype(np.dtype("u1"))

    return ret_data


def encode_data(data_dict):
    '''Dynamically find columns that fit the max_features criteria and
    encode them. In summary this function performs an integer transformation
    to compress the data and then does a one hot encoding on the compressed
    columns.'''
    master_results = dict()
    results = dict()

    # Add the headers to the encoded dictionary to keep track of columns
    master_results["Col_Names"] = data_dict["headers"]

    # Initialize stats for each columns
    for col_stat in range(data_dict["data"].shape[1]):
        results[col_stat] = {
            'allow': set(),
            'deny': set(),
            'drop': set(),
            'reset-both': set()
        }

    # Get count of different values for each action for each row
    indx_cnt = 0
    for row in range(len(data_dict["data"])):
        for indx in range(len(data_dict["data"][row])):
            if data_dict["target"][row] == 0:
                results[indx_cnt]['allow'].add(data_dict["data"][row][indx])
            if data_dict["target"][row] == 1:
                results[indx_cnt]['deny'].add(data_dict["data"][row][indx])
            if data_dict["target"][row] == 2:
                results[indx_cnt]['drop'].add(data_dict["data"][row][indx])
            if data_dict["target"][row] == 3:
                results[indx_cnt]['reset-both'].add(
                    data_dict["data"][row][indx]
                )

            indx_cnt += 1
        indx_cnt = 0

    master_results["Column_Data"] = results

    # Go through the results and find columns where the number of features of
    # deny+drop+resetboth combined is less than the number of max features
    # cols_to_encode = []
    cols_to_encode = []

    '''
    for col in range(len(results)):
        # These sums are literally only to prevent going over the 80 char
        # pep8 limit
        sum_1 = len(results[col]["deny"]) + len(results[col]["drop"])
        sum_total = sum_1 + len(results[col]["reset-both"])
        if sum_total < max_features:
            cols_to_encode.append(col)
    '''

    # Encode port columns too
    port_cols = [1, 3]

    # Find top percentage of values from port columns
    def _get_port_mapping(col_num):
        thresh = .97
        col_values = data_dict["data"][:, col_num].tolist()
        # The number that represents the thresh value of the data
        percent_cnt_val = len(col_values) * thresh

        # Create a dictionary of values and it's counts
        val_count = dict()
        for val in col_values:
            if val not in val_count:
                val_count[val] = 1
            else:
                val_count[val] += 1

        # Loop through the dictionary and add the top values until we reach
        # our threshold percentage
        curr_percent = 0.0
        dict_index = 0
        ret_dict = dict()
        while curr_percent < thresh:
            new_max = max(val_count, key=val_count.get)
            # Add key to dictionary
            ret_dict[dict_index] = new_max
            # Add percentage value to curr_percent
            curr_percent += val_count[new_max] / percent_cnt_val
            # Remove values from dictionary
            del val_count[new_max]
            # Increment Dict Index
            dict_index += 1

        return ret_dict

    master_results["Encoded_Columns"] = cols_to_encode

    # Start the encoding process
    mapping = dict()
    for col in cols_to_encode:
        # Look into the results and creating a mapping
        _vals = set()
        dict_indx = 0
        # Get the values for non allow actions
        _vals.update(results[col]["deny"])
        _vals.update(results[col]["drop"])
        _vals.update(results[col]["reset-both"])

        mapping[col] = dict()
        for i in _vals:
            mapping[col][i] = dict_indx
            dict_indx += 1

    # Port mapping
    for col in port_cols:
        mapping[col] = _get_port_mapping(col)
        cols_to_encode.append(col)

    master_results["Encoded_Col_Mapping"] = mapping

    # Transform data using encoding
    encoded_data = []
    for row in range(len(data_dict["data"])):
        new_row = []
        for i in range(len(data_dict["data"][row])):
            # len(data_dict["data"][row]) = 11
            if i in cols_to_encode:
                if (data_dict["data"][row][i] in mapping[i].values()):
                    # Append mapped value
                    new_row.append(
                        list(
                            mapping[i].values()).index(
                                data_dict["data"][row][i]
                        )
                    )

                else:
                    # The other column
                    new_row.append(len(mapping[i].keys()) + 1)

            else:
                # Append the value if it's not part of the mapping
                new_row.append(data_dict["data"][row][i])

        encoded_data.append(new_row)

    # Turn the encoded data into an array before delivering
    encoded_data = np.asarray(encoded_data)

    master_results["Integer_Encoded_Data"] = encoded_data

    # Use a column transformer to help with One Hot Encoding
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto'),
          cols_to_encode)],
        remainder='drop'
    )

    # Fit and transform the data
    ohe_data = ct.fit_transform(encoded_data).toarray()

    # Get names of columns that we encoded
    master_results["Encoded_Col_Names"] = [
        i for j,
        i in enumerate(master_results[
            "Col_Names"
        ]) if j in master_results[
            "Encoded_Columns"]
    ]

    # Get the names of the the columns that were one hot encoded
    ohe_col_names = ct.get_feature_names_out(master_results["Col_Names"])

    # Strip the "one_hot_encoder__" prefix from the column names
    for name in range(len(ohe_col_names)):
        ohe_col_names[name] = ohe_col_names[name].replace(
            "one_hot_encoder__", "")

    master_results["ohe_col_names"] = ohe_col_names.tolist()

    master_results["OHE_Data"] = ohe_data

    return master_results


def transform_and_scale(encoded_dict):
    """Takes the encoded dictionary output from ecode_data function and
    performs a transformation on the data. It will scale the data that was
    not OneHotEncoded and add the OneHoteEncoded Columns back to it."""
    # Get the continous data and drop the One Hot Encoded Data
    ret_dict = encoded_dict

    # Extract only the continous data
    cont_data = np.delete(
        ret_dict["Integer_Encoded_Data"],
        ret_dict["Encoded_Columns"],
        1,
    )

    # Get the names of the continous column values
    ret_dict["Continous_Data"] = cont_data
    ret_dict["Cont_Col_Names"] = [
        i for j,
        i in enumerate(ret_dict[
            "Col_Names"
        ]) if j not in ret_dict[
            "Encoded_Columns"]
    ]

    # Scale the conitnous data
    scaler = StandardScaler()
    ret_dict["Continous_Data"] = scaler.fit_transform(
        ret_dict["Continous_Data"]
    )

    # Combine the Scaled Continous data and the OneHotEncoded data
    ret_dict["Combined_Data"] = np.concatenate(
        (ret_dict["Continous_Data"], ret_dict["OHE_Data"]), axis=1
    )

    # Create a combined column name
    ret_dict["Combined_Col_Names"] = ret_dict[
        "Cont_Col_Names"] + ret_dict["ohe_col_names"]

    return ret_dict


if __name__ == "__main__":
    # Grab arguments
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-f", "--file", help="Path to log2.csv file", metavar='\b')
    args = parser.parse_args()

    data_path = ''

    # Validate the file path exists
    if args.file is not None:
        # Validate paths
        if os.path.exists(args.file):
            # validate blank argument that might default to the home dir
            if len(args.file) > 1:
                data_path = args.file
            else:
                raise Exception("-f argument cannot be blank...")
        else:
            raise Exception("Path Provided does not exists or is invalid...")

    data_path = "../Datasets/log2.csv"  # For Spyder Debugging

    print("Loading Data...")
    log2_data = load_log2_data(data_path)

    # Assign data
    data = log2_data["data"]
    target = log2_data["target"]

    # Encode data
    print("Encoding selected Columns...")
    # encoded_info = encode_data(log2_data, max_features=11)
    encoded_info = encode_data(log2_data)

    # Transform the data to scale and combine
    encoded_info = transform_and_scale(encoded_info)

    # Assign the combined data that we will use to train our models
    transfrmd_data = encoded_info["Combined_Data"]

    # Scale down the size of the array to better fit into memory
    transfrmd_data = transfrmd_data.astype(dtype=np.dtype("f4"))

    df = pd.DataFrame(
        transfrmd_data, columns=encoded_info["Combined_Col_Names"])
    df['Action'] = target

    # Drop Source Ports
    df = df.drop(["Source Port", "NAT Source Port"], axis=1)

    # -- Start grid search testing
    # Get the target
    df_target = df["Action"]
    # Create df without the target column
    df_data = df.drop(["Action"], axis=1)

    # Create params for svm
    svm_params = {
        SVC: {
            'C': [.000001, .00001, .0001, .001, .01, .1, 1.0, 5, 10, 20]
        }
    }

    # Create params for sgd
    sgd_params = {
        SGDClassifier: {
            'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
                      1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        }
    }

    # Run Test for SVM
    svm_search = BFMSearch()
    svm_search.run(
        svm_params, df_data, df_target, metric="weighted", create_csv=True
    )

    # Run Test for SGD
    sgd_search = BFMSearch()
    sgd_search.run(
        sgd_params, df_data, df_target, metric="weighted", create_csv=True
    )
