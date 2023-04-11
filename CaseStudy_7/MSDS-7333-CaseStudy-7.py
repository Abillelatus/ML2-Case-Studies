# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:00:28 2023

@author: Ryan Herrin, Luke Stodgel
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score
import time
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import argparse
import warnings


warnings.filterwarnings(
    "ignore",
    message="The default value of regex will change"
)

program_desc = """
Case Study 7. Minimize the monetary loss of your customer.
You may use any model(s) at your disposal.
Model explainability is NOT a factor.
"""


def format_x_data(dataframe):

    # replace the $ symbol in column 'x37' with an empty string
    dataframe['x37'] = dataframe['x37'].str.replace('$', '')
    # replace the $ symbol in column 'x32' with an empty string
    dataframe['x32'] = dataframe['x32'].str.replace('%', '')

    # Separate out the target variable
    X = dataframe.drop('y', axis=1)

    # ## Create categorical DF for imputing the mode and then one-hot encoding
    # create a list of column indices to keep
    categorical_cols = ['x24', 'x29', 'x30']

    # create a list of all column names in the DataFrame
    all_cols = dataframe.columns.tolist()

    # create a list of column names to drop
    cols_to_drop = [col for col in all_cols if col not in categorical_cols]

    # drop the columns from the DataFrame and keep the original column headers
    categorical_df = dataframe.drop(columns=cols_to_drop)

    # create an imputer object for categorical variables using mode imputation
    imputer = SimpleImputer(strategy='most_frequent')

    # fit and transform the imputer on the categorical columns
    categorical_df = imputer.fit_transform(categorical_df)

    # convert the imputed array back to a DataFrame with column headers
    categorical_df = pd.DataFrame(categorical_df, columns=categorical_cols)

    # perform one-hot encoding using the categorical_columns from before
    # adding drop_first=True increased our accuracy by .002
    # most likely due to reducing correlation problems
    df_encoded = pd.get_dummies(categorical_df, columns=categorical_df.columns,
                                drop_first=True)
    # end

    # ## Create numeric df for knnimputation
    # drop non-float columns from original DataFrame
    df_float = X.drop(dataframe.columns[[24, 29, 30]], axis=1)

    # instantiate the KNN imputer with k=3
    imputer = KNNImputer(n_neighbors=3)

    # impute the missing values in the DataFrame
    df_imputed = pd.DataFrame(imputer.fit_transform(df_float),
                              columns=df_float.columns)
    # end

    # ## Concatenate and scale
    # combine one-hot encoded DataFrame with float DataFrame
    x_df_final = pd.concat([df_imputed, df_encoded], axis=1)

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x_df_final)

    # Convert scaled NumPy array back to Pandas dataframe with column headers
    x_df_scaled = pd.DataFrame(X_scaled,
                               columns=x_df_final.columns).astype('float32')

    return x_df_scaled


def get_best_threshold(predictions, target, show_chart=True):
    # Metric container
    metrics = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])
    # prec_m_score = []
    acc_m_score = []
    # recall_m_score = []
    # f1_m_score = []

    # Run through the thresholds and extract the scores
    for thrshlds in range(1, 100, 1):
        thrshld = thrshlds / 100
        tmp_y_pred = predictions >= thrshld
        tmp_y_pred = np.where(tmp_y_pred, 1, 0)

        # Get and append metrics
        tmp_metric = []
        tmp_metric.append(accuracy_score(target, tmp_y_pred))
        tmp_metric.append(precision_score(target, tmp_y_pred))
        tmp_metric.append(recall_score(target, tmp_y_pred))
        tmp_metric.append(f1_score(target, tmp_y_pred))

        acc_m_score.append(accuracy_score(target, tmp_y_pred))

        # Append to metric container
        metrics.loc[len(metrics)] = tmp_metric

    ax = metrics.plot(figsize=(12, 5), title='Prediction Threshold Scores')
    # Sets the y-axis to be based on a percentage
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda target, _: '{:.0%}'.format(target))
    )
    ax.set_xlabel("Threshold Amount (as percentage)")

    return acc_m_score.index(max(acc_m_score)) / 100


def build_and_compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def run_dense_nn(x_data, y_data):

    # define k-fold cross validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)

    # initialize arrays to store the predictions for each fold
    fold_preds = np.zeros(y_data.shape)
    val_loss = []
    val_accuracy = []

    # iterate over each fold
    for i, (train_index, test_index) in enumerate(kf.split(x_data)):

        # get the train and test data for this fold
        x_train, x_test = x_data[train_index], x_data[test_index]

        y_train, y_test = y_data[train_index], y_data[test_index]

        # build and compile the model
        my_model = build_and_compile_model()

        # define early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5
        )

        # train the model
        history = my_model.fit(
            x_train, y_train,
            epochs=1000,
            batch_size=100,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping]
        )

        # evaluate the model on test set
        y_pred = my_model.predict(x_test)
        # store predictions for this fold
        fold_preds[test_index] = y_pred.flatten()
        '''
        # plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.show()
        '''
        # add the validation loss and accuracy values to the lists
        val_loss.append(history.history['val_loss'][-1])
        val_accuracy.append(history.history['val_accuracy'][-1])

    # calculate the average validation loss and accuracy
    avg_val_loss = sum(val_loss) / len(val_loss)
    avg_val_accuracy = sum(val_accuracy) / len(val_accuracy)

    print("Average validation loss across all folds: ", avg_val_loss)
    print("Average validation accuracy across all folds: ", avg_val_accuracy)

    # Plot mean metrics
    # Container used to feed into charts
    mean_validation_values = []

    mean_validation_values.append(np.mean(avg_val_loss))
    mean_validation_values.append(np.mean(avg_val_accuracy))

    # Names for the charts
    val_metric_names = ['Avg_val_loss', 'Avg_val_acc']

    # Create the barplot figure
    fig, ax = plt.subplots()
    bars = ax.barh(val_metric_names, mean_validation_values)
    ax.bar_label(bars)
    plt.yticks(val_metric_names)
    plt.title(
        'Metrics for our Final Dense NN Model')

    plt.xlabel('Percentage', fontsize=11, color='blue')
    plt.ylabel('Metrics', fontsize=11, color='blue')
    plt.show()
    # end

    # calculate accuracy, precision, recall, and f1_score for all data
    threshold = get_best_threshold(fold_preds, y_data, True)
    print("best threshold: ", threshold)
    y_pred_classes_all = (fold_preds > threshold).astype(int)
    accuracy_all = accuracy_score(y_data, y_pred_classes_all)
    precision_all = precision_score(y_data, y_pred_classes_all)
    recall_all = recall_score(y_data, y_pred_classes_all)
    f1_all = f1_score(y_data, y_pred_classes_all)

    print("Accuracy on all data:", accuracy_all)
    print("Precision on all data:", precision_all)
    print("Recall on all data:", recall_all)
    print("F1 score on all data:", f1_all)

    # Plot mean metrics
    # Container used to feed into charts
    mean_values = []

    mean_values.append(np.mean(accuracy_all))
    mean_values.append(np.mean(precision_all))
    mean_values.append(np.mean(recall_all))
    mean_values.append(np.mean(f1_all))

    # Names for the charts
    names = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Create the barplot figure
    fig, ax = plt.subplots()
    bars = ax.barh(names, mean_values)
    ax.bar_label(bars)
    plt.yticks(names)
    plt.title(
        'Metrics for our Final Dense NN Model')

    plt.xlabel('Percentage', fontsize=11, color='blue')
    plt.ylabel('Metrics', fontsize=11, color='blue')
    plt.show()
    # end

    # Make confusion matrix and calculate total cost
    cm = confusion_matrix(y_data, y_pred_classes_all)

    tn, fp, fn, tp = cm.flatten()
    false_positives_cost = fp * 35
    false_negatives_cost = fn * 15
    total_cost = false_negatives_cost + false_positives_cost
    print("Total cost = ", total_cost)

    # cm = confusion_matrix(y_data, y_pred_classes_all)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', square=True)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.show()
    # end

    return total_cost


if __name__ == '__main__':
    # Grab arguments
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-f", "--file", help="Path to final_project.csv file", metavar='\b')
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

    # data_path = '../Datasets/final_project.csv'  # For Spyder Debugging

    if data_path == '' or data_path is None:
        raise Exception('[Error] final_project.csv path must be'
                        ' specified with -f or --file')

    # Record the start time
    start_time = time.time()

    # Read the CSV file using pandas
    df = pd.read_csv(data_path)
    y = df['y'].astype('uint8')

    x_scaled = format_x_data(df)

    # this algorithm requires a np.array to create the train/test split
    # for k-fold predictions.
    x_scaled = x_scaled.values

    # total_cost = 0
    # for i in range(5):
    #    total_cost +=
    run_dense_nn(x_scaled, y)

    # average_cost = total_cost / 5

    # print(average_cost)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in seconds
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
