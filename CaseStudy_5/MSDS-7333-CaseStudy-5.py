#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:53:51 2023

@author: Ryan Herrin, Luke Stodgel
"""
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, KFold
from sklearn.compose import ColumnTransformer

# Custom class
from BFM_Search import BFMSearch


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


def encode_data(data_dict, max_features):
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
    cols_to_encode = []
    for col in range(len(results)):
        # These sums are literally only to prevent going over the 80 char
        # pep8 limit
        sum_1 = len(results[col]["deny"]) + len(results[col]["drop"])
        sum_total = sum_1 + len(results[col]["reset-both"])
        if sum_total < max_features:
            cols_to_encode.append(col)

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

    master_results["Encoded_Col_Mapping"] = mapping

    # Transform data using encoding
    encoded_data = []
    for row in range(len(data_dict["data"])):
        new_row = []
        for i in range(len(data_dict["data"][row])):
            if (i in cols_to_encode):
                if (data_dict["data"][row][i] in mapping[i].keys()):
                    new_row.append(mapping[i][data_dict["data"][row][i]])
                else:
                    new_row.append(len(mapping[i].keys()) + 1)
            else:
                new_row.append(data_dict["data"][row][i])

        encoded_data.append(new_row)

    # Turn the encoded data into an array before delivering
    encoded_data = np.asarray(encoded_data)

    master_results["Integer_Encoded_Data"] = encoded_data

    # Use a column transformer to help with One Hot Encoding
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto'), cols_to_encode)],
        remainder='drop'
    )

    # Fit and transform the data
    ohe_data = ct.fit_transform(encoded_data).toarray()

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

    return ret_dict


def run_svm_cv(data, target, params):
    """Run cross validation on the SVM model and generate the results"""
    kf = KFold(n_splits=5)
    scoring = ['accuracy', 'precision_macro', 'recall_macro',
               'f1_macro']
    clf = SVC(**params)
    scores = cross_validate(
        clf,
        data,
        target,
        cv=kf,
        scoring=scoring,
        n_jobs=-1
    )

    print(
        "\n"
        "SVM Results:\n"
        "------------\n"
        "Accuracy: {}\n"
        "Precision: {}\n"
        "Recall: {}\n"
        "F1: {}\n".format(
            round(sum(scores["test_accuracy"]) / 5, 4),
            round(sum(scores["test_precision_macro"]) / 5, 4),
            round(sum(scores["test_recall_macro"]) / 5, 4),
            round(sum(scores["test_f1_macro"]) / 5, 4)
        ))

    return scores


if __name__ == "__main__":
    # --- Temp path to be replace by arguments
    data_path = "../Datasets/log2.csv"
    print("Loading Data...")
    log2_data = load_log2_data(data_path)

    # Assign data
    data = log2_data["data"]
    target = log2_data["target"]

    # Encode data
    print("Encoding selected Columns...")
    encoded_info = encode_data(log2_data, max_features=11)

    # Transform the data to scale and combine
    encoded_info = transform_and_scale(encoded_info)

    # Assign the combined data that we will use to train our models
    transfrmd_data = encoded_info["Combined_Data"]

    # Scale down the size of the array to better fit into memory
    transfrmd_data = transfrmd_data.astype(dtype=np.dtype("f4"))

    # Get Params to test
    svc_params = {
        SVC: {
            'C': np.linspace(0.00001, 20, 200).tolist(),
            'kernel': ['rbf']
        }
    }

    sgd_params = {
        SGDClassifier: {
            'alpha': np.linspace(0.00001, 20, 200).tolist(),
            'n_jobs': [-1]
        }
    }

    # SVC gridsearch
    best_svc_model = BFMSearch()
    best_svc_model.run(
        svc_params, transfrmd_data, target, metric="weighted", create_csv=True)

    # SGD Gridsearch
    best_sgd_model = BFMSearch()
    best_sgd_model.run(
        sgd_params, transfrmd_data, target, metric="weighted", create_csv=True)


# Temp break line for testing. Will delete later
# ----------------------------------------------------------------------------
    '''
    # Create SVC params
    params = {
        SVC: {
            'C': [.01, .1, 1, 5, 15, 20],
            'degree': [1, 3, 5],
            'gamma': ['scale', 'auto'],
            'kernel': ['poly', 'rbf', 'linear'],
            'class_weight': [None, 'balanced'],
            'decision_function_shape': ['ovr', 'ovo']
        }
    }

    params = {
        SGDClassifier: {
            'loss': ['log_loss', 'modified_huber', 'squared_hinge'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            # 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 1.0],
            'tol': [1e-4, 1e-3, 1e-2],
            'early_stopping': [False, True],
            'class_weight': ['balanced', None],
            'n_jobs': [-1]
        }
    }
    best_model = BFMSearch()
    best_model.run(params, data_scaled, target, metric="weighted")
    '''

    '''
    # Params
    svm_params = {
        'C': 20,
        'degree': 3,
        'gamma': 'scale',
        'kernel': 'poly',
        'class_weight': 'balanced',
        'decision_function_shape': 'ovo'
    }

    # Run the model and generate results
    print("Running SVM Model...")
    svm_scores = run_svm_cv(data_scaled, target, svm_params)
    '''





























