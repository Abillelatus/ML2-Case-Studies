# -*- coding: utf-8 -*-
"""
Created on  Mar 13 19:49:00 2023

@author: Ryan Herrin, Luke Stodgel
"""

# Build a dense neural network to accurately detect the particle.
# The goal is to maximize your accuracy.
# Include a discussion of how you know your model has finished training as
# well as what design decisions you made while building the network.

import os
import argparse
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

program_desc = """
Case Study 6. Build a dense neural network to accurately detect the particle.
"""


def build_fit_predict_and_output_results(x_train, x_test, y_train, y_test):

    # sequential type of model
    my_model = tf.keras.Sequential()
    # input layer
    my_model.add(tf.keras.Input(shape=(28,)))
    # first hidden layer
    my_model.add(tf.keras.layers.Dense(512, activation='relu'))
    # second layer
    my_model.add(tf.keras.layers.Dense(512, activation='relu'))
    # third layer
    my_model.add(tf.keras.layers.Dense(512, activation='relu'))
    # for regularization
    my_model.add(tf.keras.layers.Dropout(0.3))
    # output layer (matches up with our target) sigmoid for binary
    # classification
    my_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # since we are doing a regression problem, we need a regression loss,
    # continuous problem, continuous loss.
    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5
    )

    # train the model
    history = my_model.fit(
        x_train, y_train,
        # epochs=5,
        epochs=1000,
        batch_size=1000,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping]
    )

    # evaluate the model on test set
    y_pred = my_model.predict(x_test)
    y_pred_classes = (y_pred > .5).astype(int)

    # calculate precision, recall, and f1_score
    accuracy = round(accuracy_score(y_test, y_pred_classes), 3)
    precision = round(precision_score(y_test, y_pred_classes), 3)
    recall = round(recall_score(y_test, y_pred_classes), 3)
    f1 = round(f1_score(y_test, y_pred_classes), 3)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)

    # Make confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', square=True)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.show()

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

    # Container used to feed into charts
    mean_values = []

    mean_values.append(np.mean(accuracy))
    mean_values.append(np.mean(precision))
    mean_values.append(np.mean(recall))
    mean_values.append(np.mean(f1))

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

    # Create list of lists for the boxplot
    values = [accuracy, precision, recall, f1]

    return my_model, values


def load_csv(csv_path):
    """Load in combined csv file if generated previously"""
    try:
        return pd.read_csv(csv_path)

    except Exception as err:
        print(str(err))


if __name__ == '__main__':
    # Grab arguments
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-f", "--file", help="Path to all_train.csv file", metavar='\b')
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

    # data_path = '../datasets/all_train.csv'  # For Spyder Debugging
    if data_path == '':
        raise Exception('[Error] log2.csv path must be'
                        ' specified with -f or --file')

    # read in the csv
    # path = '.\\all_train\\all_train.csv'
    # data_path = '../Datasets/all_train.csv'

    df = load_csv(data_path)

    # create X and y variables
    X = df.drop('# label', axis=1).astype("float32")
    y = df['# label'].astype("float32")

    # create a StandardScaler object
    scaler = StandardScaler()
    # fit the scaler to your data
    X_scaled = scaler.fit_transform(X)

    # create train test split4
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2
    )

    # build the model in preparation for fitting
    my_model, values = build_fit_predict_and_output_results(
        x_train, x_test, y_train, y_test
    )
