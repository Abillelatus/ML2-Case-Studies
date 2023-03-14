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
from sklearn.model_selection import cross_validate, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics as mt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from IPython import get_ipython

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


def get_model_results_and_plots(
        dataframe, model, best_params, thresh, num_tests, model_name, e_data):
    '''Common function to get results from a passed model'''

    # Remove the 'target' column from the DataFrame and assign the result to X
    X = dataframe.drop('Action', axis=1).astype('float32')
    # Assign the 'target' column to y
    y = dataframe['Action']

    # Names for the charts
    names = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Container to hold the means
    accuracy_means = []
    precision_means = []
    recall_means = []
    f1_means = []

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
        accuracy_means.append(
            round(accuracy_score(y_test, y_hat), 3))
        precision_means.append(
            round(precision_score(y_test, y_hat, average='weighted',
                                  zero_division=0), 3))
        recall_means.append(
            round(recall_score(y_test, y_hat, average='weighted'), 3))
        f1_means.append(
            round(mt.f1_score(y_test, y_hat, average='weighted'), 3))

    # --Feature importances begin -- #
    if hasattr(test_model, 'coef_'):
        # Get feature importances
        importances = test_model.coef_[0]
        # Get column names
        feature_names = X.columns
        # Create dictionary of feature names and importances
        feature_dict = dict(zip(feature_names, importances))
        # Sort features by importance in descending order
        sorted_features = sorted(
            feature_dict.items(), key=lambda x: x[1], reverse=True
        )
        # Print sorted feature importances
        print("Feature Importances:")
        for feature, importance in sorted_features[:10]:
            ft = feature
            # Map back the actual port number for NAT Destination
            if "NAT" in ft:
                ft = ft.split('_')
                ft[-1] = e_data["Encoded_Col_Mapping"][3][int(ft[-1])]
                feature = ft[0] + "_" + str(ft[-1])
            # Map back the actual port num for Destination port
            elif "Destination" in ft:
                ft = ft.split('_')
                ft[-1] = e_data["Encoded_Col_Mapping"][1][int(ft[-1])]
                feature = ft[0] + "_" + str(ft[-1])
            print(feature, ": ", importance)
    else:
        print("Feature importances are not available for this model.")
    # -- Feature importances end -- #

    mean_values.append(np.mean(accuracy_means))
    mean_values.append(np.mean(precision_means))
    mean_values.append(np.mean(recall_means))
    mean_values.append(np.mean(f1_means))

    # -- Confusion matrix heatmap begin -- #
    # Define dictionary to map integer labels to string labels
    label_map = {0: 'allow', 1: 'deny', 2: 'drop', 3: 'reset-both'}
    # Convert integer labels to string labels for y_test
    y_test_labels = [label_map[label] for label in y_test]
    # Convert integer labels to string labels for y_hat
    y_hat_labels = [label_map[label] for label in y_hat]
    # Update classes to contain string labels
    classes = ['allow', 'deny', 'drop', 'reset-both']
    # Create confusion matrix
    confusion_mat = confusion_matrix(
        y_test_labels, y_hat_labels, labels=classes
    )
    # Convert to pandas DataFrame
    confusion_df = pd.DataFrame(confusion_mat, index=classes, columns=classes)
    # Create heatmap
    sns.heatmap(confusion_df, annot=True, cmap="Blues", fmt="g")

    # Add labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("{} | Confusion Matrix".format(model_name))
    # Display plot
    plt.show()
    # -- Confusion matrix heatmap end -- #

    # Create the barplot figure
    fig, ax = plt.subplots()
    bars = ax.barh(names, mean_values)
    ax.bar_label(bars)
    plt.yticks(names)

    # Seperate titles if using threshold or not
    if thresh is not None:
        plt.title(
            '{} |'
            ' Average Mean Scores |'
            ' ProbThreshold = {} |'
            ' Num_Runs={}'.format(model_name, threshold, num_tests))
    else:
        plt.title(
            '{} |'
            ' Average Mean Scores |'
            ' Num_Runs={}'.format(model_name, num_tests))

    plt.xlabel('Percentage', fontsize=11, color='blue')
    plt.ylabel('Metrics', fontsize=11, color='blue')
    plt.show()

    # Create list of lists for the boxplot
    values = [accuracy_means, precision_means, recall_means, f1_means]

    # Create Boxplot of model metric variances
    bx_plt = plt.figure()
    bx_plt.suptitle('{} | Model Metric Variances over {} runs'.format(
        model_name, num_tests
    ))
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

    print('Accuracy:  Mean = {} | Variance = {}'.format(acc_mean, acc_var))
    print('Precision: Mean = {} | Variance = {}'.format(prec_mean, prec_var))
    print('Recall:    Mean = {} | Variance = {}'.format(rec_mean, rec_var))
    print('F1_Score:  Mean = {} | Variance = {}'.format(f1_mean, f1_var))
    print()


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
            if len(args.file) > 2:
                data_path = args.file
            else:
                raise Exception("-f argument cannot be blank...")
        else:
            raise Exception("Path Provided does not exists or is invalid...")

    # data_path = "../Datasets/log2.csv"  # For Spyder Debugging

    if data_path == '':
        raise Exception('[Error] log2.csv path must be'
                        ' specified with -f or --file')

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

    # Get the best params for SVM
    best_params_SVM = {
        'C': 1, 'kernel': 'rbf',
        'class_weight': None, 'decision_function_shape': 'ovr'
    }

    # Get the best params for SGD
    best_params_sgd = {
        'loss': 'modified_huber', 'penalty': 'l2', 'alpha': 1e-6,
        'max_iter': 1000, "learning_rate": 'optimal',
        'early_stopping': True, 'n_iter_no_change': 5,
        'class_weight': None
    }

    # SVM model using predict()
    print("Creating SVM...")
    model_metrics_svm_trad = get_model_results_and_plots(
        df, SVC, best_params_SVM,
        None, 1, 'SVC', encoded_info
    )

    # SGD model using predict()
    print("Creating SGD model...")
    model_metrics_sgd_trad = get_model_results_and_plots(
        df, SGDClassifier, best_params_sgd, None, 1,
        'SGD', encoded_info
    )

    print('\n')  # Seperate log from results

    # Display the results on Console
    print("SVM")
    print("-----------------------------")
    display_model_metrics(model_metrics_svm_trad)
    print("SGD")
    print("------------------------------")
    display_model_metrics(model_metrics_sgd_trad)

    print("Script Complete...")
