#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:13:47 2023

@author: Luke Stodgel, Ryan Herrin
"""
import os
import argparse
import pandas as pd


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


if __name__ == "__main__":
    # Grab arguments if provided.
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-u", "--unique_csv", help="Path to the train.csv", metavar='\b')
    parser.add_argument(
        "-t", "--train_csv", help="Path to the unique_m.csv", metavar='\b')
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose logging", action="store_true")
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
        input_err = ("One or more input files were found. Please specify unique"
                     " and train csv files using the -u and -t flags...")

        raise Exception(input_err)

    # Start the main script
    # Read in data
    unique_df, train_df = read_in_date(unique_loc, train_loc)

    # Create a dataframe based on joining the two dataframes
    combined_df = join_unique_and_train(unique_df, train_df)

























