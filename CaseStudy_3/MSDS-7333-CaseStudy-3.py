#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 08:49:00 2023

@author: Ryan Herrin, Luke Stodgel
"""

import re
import os
import time
import email
import pickle
import argparse
import pandas as pd
import numpy as np
from io import StringIO
import multiprocessing as m_proc
from multiprocessing import Pool
from html.parser import HTMLParser
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import GridSearchCV  # KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics as mt


# Description for program
program_desc = '''Program to train a model to identify email spam that could
later be used as a filter.
'''

# Symbol mapping for processing the payload data
symbl_map = {
    '!': 's_expl', '@': 's_at', '#': 's_pound', '$': 's_dollar',
    '%': 's_percent', '^': 's_carrot', '&': 's_amp', '*': 's_astr',
    '(': 's_l_prnth', ')': 's_r_prnth', '-': 's_dash',
    '_': 's_under_score', '=': 's_equal', '+': 's_plus',
    '`': 's_back_quote', '~': 's_tilda', '{': 's_open_brace',
    '}': 's_close_brace', '[': 's_open_bracket', ']': 's_close_bracket',
    '|': 's_pipe', '\\': 's_b_slash', ':': 's_colon', ';': 's_semi_colon',
    "'": 's_single_quote', '"': 's_double_quote', '<': 's_left_arrow',
    '>': 's_right_arrow', '?': 's_q_mark', ',': 's_comma', '.': 's_period',
    '/': 's_forward_slash'}


class EmailProcessing:
    '''Class that parses through emails and attempts to return a dataframe
    of emails broken down into features that could be used to be fed into a
    machine learning process.

    Example Invocation:
    ---------------------
    example = EmailProcessing()
    example.process('../path/to/email_dirs')'''

    def __init__(self):
        self.path = None
        self.file_paths = []
        self.email_content = dict()
        self.stats = dict()
        self.pool_slices = None
        self.cpu_core_count = m_proc.cpu_count()
        self.combined_data = None
        self.static_col_names = None
        self.email_df = None

    class MLStripper(HTMLParser):
        '''Inner class to handle html parsing/stripping of some emails'''

        def __init__(self):
            super().__init__()
            self.reset()
            self.strict = False
            self.convert_charrefs = True
            self.text = StringIO()

        def handle_data(self, d):
            self.text.write(d)

        def get_data(self):
            return self.text.getvalue()

    def _log(self, x):
        '''Logging formatter'''
        try:
            print('[EmailProcessing] > {}'.format(str(x)))
        except Exception as err:
            print(err)

    def _build_file_path_list(self):
        '''Walk through the root folder and build a list of absolute paths to
        the emails'''
        ignore = ['.py', '.ipynb', '.tmp', 'cmds']  # Files to ignore
        for root, dirs, files in os.walk(self.path):
            for name in files:
                # Make sure we are not bringing in non email files
                # if any(ext not in ignore for ext in os.path.join(
                #        root, name)):
                if name not in ignore:
                    self.file_paths.append(os.path.join(root, name))
                else:
                    pass

    def _process_html(self, html_file):
        '''If HTML file, strip tags and return data'''
        # Create the MLStripper object and send it to process
        strip_fx = self.MLStripper()
        strip_fx.feed(html_file)

        return strip_fx.get_data()

    def _read_email_contents(self, file_pool):
        '''Attempt to read the contents of an email and store the data as a
        dictionary where:
            {
                str(filename) : [
                    int(is_spam), str(path), str(payload), dict(stats)
                    ]
            }
        '''
        email_content = dict()
        not_spam_kwd = '_ham'
        is_spam = 0  # 0 - Not Spam, 1 - Is Spam
        file_name = ''

        # Attempt to read in the file
        try:
            for file in file_pool:
                with open(file, 'rb') as infile:
                    msg = email.message_from_binary_file(infile)
                    # Exatract information from email
                    for part in msg.walk():
                        data = part.get_payload()

                        # Get the file name
                        if '\\' in file:
                            file_name = file.split('\\')[-1]
                        else:
                            file_name = file.split('/')[-1]

                        # Determine from the file path if the messages is
                        if not_spam_kwd not in file:
                            is_spam = 1

                        # Do any pre processing to the payload data if needed
                        data = self._process_payload(data)

                        if data == 1:
                            pass
                        else:
                            # Initialize an empty payload stats dictionary
                            init_stat_dict = {'stats': dict()}

                            # Add to the dictionary
                            email_content[file_name] = [
                                is_spam,
                                file,
                                data,
                                init_stat_dict]

        except Exception as err:
            print(str(err))

        return email_content

    def _process_payload(self, payload_data):
        '''Process the payload before passing it back.'''
        _data = payload_data

        if isinstance(_data, list):
            return 1

        else:
            # If it's HTML data, strip the tags and return only the data
            if '<html>' in _data.lower():
                _data = self._process_html(_data)

        return _data

    def _create_payload_stats(self, email_dict):
        '''Parse through the data and generate different stats from the payload
        including Total Word Count, Total Symbol Count, How many words have the
        first letter capitalized, how many words are all caps, average line
        length, and a dictionary count of all symbols and words.'''
        # Stats value is located at email[3]['stats']
        # Payload to examine is email[2]
        emails = email_dict

        for entry in emails:
            total_words = 0   # Total number of words
            total_symbols = 0  # Total number of symbols
            first_letter_cap = 0  # Num of Words with first letter capitalized
            all_cap_words = 0  # Num of words that have words with all caps
            avg_line_len = 0
            line_count = 0
            symbol_count = dict()
            word_count = dict()

            # Get payload data
            payload_str = emails[entry][2]

            wrk_pyld = payload_str.split('\n')

            for line in range(len(wrk_pyld)):
                # Strip leading and trailing white space
                wrk_pyld[line] = wrk_pyld[line].strip()

                # Ignore blank lines
                if len(wrk_pyld[line]) >= 1:
                    # Add line length to avg_line_len
                    avg_line_len += len(wrk_pyld[line])
                    line_count += 1
                    # Remove apostrophe to avoid acceidently excluding
                    # certain words
                    wrk_pyld[line] = wrk_pyld[line].replace("'", "")
                    # Get num of words, removing all symbols and numbers
                    words = list([wrd for wrd in re.sub(
                        r'[\W_]+', ' ',
                        wrk_pyld[line]).split(' ') if wrd.isalpha()])

                    # Get all the words in the payload
                    for word in range(len(words)):
                        # Increment the count of words if it's not blank
                        if len(words[word]) > 0:
                            total_words += 1
                            # Check capitalization and add to stats
                            if words[word][0].isupper():
                                # The whole word is capitalized
                                if words[word].isupper():
                                    # The whole word need to be longer than
                                    # 1 char to avoid "I" words
                                    if len(words[word]) > 1:
                                        all_cap_words += 1
                                else:
                                    # If only the first letter is capitalized
                                    first_letter_cap += 1

                            # Add to the dictionary of words
                            # Check if word is longer than len 15, and create
                            # It's own dictionary for that
                            min_word_len = 4
                            max_word_len = 12
                            if len(words[word]) > max_word_len:
                                if 'large_str' in word_count:
                                    word_count['large_str'] += 1
                                else:
                                    word_count['large_str'] = 1
                            # Check for min word length
                            elif len(words[word]) >= min_word_len:
                                if words[word].lower() in word_count:
                                    word_count[words[word].lower()] += 1
                                else:
                                    word_count[words[word].lower()] = 1

                    # Get all the symbols in the payload
                    for char in wrk_pyld[line]:
                        if not char.isalpha() and not char.isnumeric():
                            # Remove blank characters
                            if char != ' ' and char != '':
                                # Add to total symbol count
                                total_symbols += 1
                                # Add to dictionary using the symbol mapping
                                try:
                                    if symbl_map[str(char)] in symbol_count:
                                        symbol_count[symbl_map[str(char)]] += 1
                                    else:
                                        symbol_count[symbl_map[str(char)]] = 1
                                # Handle all other symbols and chars
                                except Exception as err:
                                    err = err
                                    if 's_other' in symbol_count:
                                        symbol_count['s_other'] += 1
                                    else:
                                        symbol_count['s_other'] = 1

            # Calculate average line length
            if line_count == 0:
                avg_line_len = 0
            else:
                avg_line_len = round((avg_line_len / line_count), 0)

            # Add stats to the dictionary
            emails[entry][3]['stats']['total_words'] = total_words
            emails[entry][3]['stats']['total_symbols'] = total_symbols
            emails[entry][3]['stats']['first_letter_cap'] = first_letter_cap
            emails[entry][3]['stats']['all_cap_words'] = all_cap_words
            emails[entry][3]['stats']['avg_line_len'] = avg_line_len
            emails[entry][3]['stats']['line_count'] = line_count
            emails[entry][3]['stats']['symbol_count'] = symbol_count
            emails[entry][3]['stats']['word_count'] = word_count

        return emails

    def _prepare_dict_for_df(self, e_data_dict):
        '''Prepares the dictionary to be more easily integrated with the
        delivered dataframe'''
        ret_dict = {}
        # Start with static columns that should appear first
        _static_col_names = [
            'line_count', 'avg_line_len',
            'total_words', 'total_symbols', 'all_cap_words',
            'first_letter_cap']

        dict_i = 0  # Counter for appending

        for key in e_data_dict.keys():
            _tmp_dict = dict()
            _tmp_dict['filename'] = key
            _tmp_dict['is_spam'] = e_data_dict[key][0]

            for col in _static_col_names:
                _tmp_dict[col] = e_data_dict[key][3]['stats'][col]

            # Add words to the _tmp dict
            for word in e_data_dict[key][3]['stats']['word_count'].keys():
                _words = e_data_dict[key][3]['stats']['word_count']
                _tmp_dict[word] = _words[word]
            # Add symbols to the _tmp dict
            for smbl in e_data_dict[key][3]['stats']['symbol_count'].keys():
                _smbl = e_data_dict[key][3]['stats']['symbol_count']
                _tmp_dict[smbl] = _smbl[smbl]

            ret_dict[dict_i] = _tmp_dict
            dict_i += 1

        return ret_dict

    def _generate_dataframe(self, processed_payload_dicts):
        '''Iterate through the processed payload data and create a combined
        dataframe that can be used for analysis'''
        ppd = processed_payload_dicts  # Create smaller name
        combined_dict = dict()
        dict_i = 0  # Iterator for key value in combined dataframe

        # Loop through each returned dictionary and add every entry to the
        # combined_dict variable. The number of returned dictionaries is equal
        # to the number of n_jobs defined.
        for result in range(len(ppd)):
            for val in ppd[result]:
                combined_dict[dict_i] = ppd[result].get(val)
                dict_i += 1

        # Turn the list of dicts into a dataframe
        ret_df = pd.DataFrame.from_dict(combined_dict, "index")
        # Replace the nan's with 0's
        ret_df = ret_df.replace(np.nan, 0)

        # Save a copy to the object
        self.email_df = ret_df

        return ret_df

    def export_dataframe_to_binary(self):
        '''Export the dataframe to a binary file for faster debugging'''
        outfile_loc = os.getcwd() + '/email_df.bin'
        try:
            with open(outfile_loc, 'wb') as outfile:
                pickle.dump(self.email_df, outfile)

        except Exception as err:
            self._log('Error Exporting binary data...')
            self._log(str(err))

    def import_dataframe_from_binary(self, bin_file):
        '''Import a previously saved binary file'''
        try:
            with open(bin_file, 'rb') as infile:
                self.email_df = pickle.load(infile)

        except Exception as err:
            self._log('Error Importing binary data...')
            self._log(str(err))

    def _get_mp_pool_slices(self):
        '''Helper function to seperate the number of files in seperate
        lists. The number of lists depends on how mony CPU cores the local
        machine has, or what was defined'''
        pool = dict()
        for core in range(self.cpu_core_count):
            pool[core] = []

        # Seperate the files into the pools
        curr_core = 0
        for file in self.file_paths:
            pool[curr_core].append(file)
            curr_core += 1
            # Reset core count of reached the max number of cores
            if curr_core > (self.cpu_core_count - 1):
                curr_core = 0

        self.pool_slices = pool  # For object inspection

        return pool

    def _mp_executor(self, file_list):
        '''Function Execution Director for multiprocssing. Any NON-ML
        parsing / feature creating function can be added to here.'''
        # Process the emails
        email_collection = self._read_email_contents(file_list)
        # Generate different stats for each email
        email_collection = self._create_payload_stats(email_collection)
        # Create formated dictionaries that will be used to create a combined
        # dataframe
        email_data_dicts = self._prepare_dict_for_df(email_collection)

        return email_data_dicts

    def process(self, path_to_email_dirs, n_jobs=None):
        '''Iterate through the emails process the information.
        Params:
            path_to_email_dir str : absolute path to directory containing
                emails
            n_jobs int : number of cores to use.
                Default = Use all avaiable cores
        Returns:
        pandas DataFrame
        '''
        self._log('Reading Emails and generating Dataframe...')
        # Set core count if n_jobs is pass3
        if n_jobs is not None:
            if n_jobs == -1:
                self.cpu_core_count = m_proc.cpu_count()
            else:
                self.cpu_core_count = int(n_jobs)

        # Set path to email samples
        self.path = path_to_email_dirs
        # Build list of file paths to iterate through
        self._build_file_path_list()

        # Seperate the files into multiple lists to enable multiprocessing.
        # This is only for data processing and not using models to create
        # new feature columns.
        process_slices = self._get_mp_pool_slices()

        # Grab lists from the processes and add them to a pool that will
        # get passed to the multiprocessor
        process_pool = []
        for key in process_slices:
            process_pool.append(process_slices[key])

        # Run multiprocessing using the Pool API. From this point the internal
        # _mp_executer will control the pipeline of each process until a
        # dataframe is returned
        with Pool(self.cpu_core_count) as p:
            pool_results = p.map(self._mp_executor, process_pool)

        # Create a dataframe from the processed emails
        self.combined_data = self._generate_dataframe(pool_results)

        self._log("Complete...")

        return self.combined_data


def get_best_threshold(df, params, show_chart=True):
    '''Find the best threshold percentage for obtaining the fewest false
    positives. Returns Threshold value and model'''
    X = df.drop(['is_spam', 'filename'], axis=1)
    y = df['is_spam']
    # Create test train split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Create predictions using probability
    mnb = MultinomialNB(**params)
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict_proba(X_test)  # [:, 1] >= .50

    # Metric container
    metrics = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])
    prec_m_score = []

    # Run through the thresholds and extract the scores
    for thrshlds in range(1, 100, 1):
        thrshld = thrshlds / 100
        tmp_y_pred = y_pred[:, 1] >= thrshld
        tmp_y_pred = np.where(tmp_y_pred, 1, 0)

        # Get and append metrics
        tmp_metric = []
        tmp_metric.append(accuracy_score(y_test, tmp_y_pred))
        tmp_metric.append(precision_score(y_test, tmp_y_pred))
        tmp_metric.append(recall_score(y_test, tmp_y_pred))
        tmp_metric.append(f1_score(y_test, tmp_y_pred))

        prec_m_score.append(precision_score(y_test, tmp_y_pred))

        # Append to metric container
        metrics.loc[len(metrics)] = tmp_metric

    ax = metrics.plot(figsize=(12, 5), title='Prediction Threshold Scores')
    # Sets the y-axis to be based on a percentage
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y))
    )
    ax.set_xlabel("Threshold Amount (as percentage)")

    return prec_m_score.index(max(prec_m_score)) / 100


def get_best_params(w_dataframe):
    # Split the data into X and y
    X_scaled = w_dataframe.drop(['is_spam', 'filename'], axis=1)
    y = w_dataframe['is_spam']

    # Create standard range
    # Define the parameter grid to search over
    param_grid = {
        'alpha': [0.01, 0.1, 1.0],
        'fit_prior': [True, False]
    }

    # Create a MultinomialNB classifier
    mnb = MultinomialNB()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(mnb,
                               param_grid,
                               scoring='precision',
                               cv=5)

    # Fit the grid search to the data
    grid_search.fit(X_scaled, y)

    return(grid_search.best_params_)


def get_NB_results_and_plots(dataframe, best_params, thresh, num_tests):
    '''Get the mse of the lasso model with multiple runs'''
    # Split the data into train and test sets
    X = dataframe.drop(['is_spam', 'filename'], axis=1)
    y = dataframe['is_spam']

    # Names for the charts
    names = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Container to hold the means
    mnb_accuracy_means = []
    mnb_precision_means = []
    mnb_recall_means = []
    mnb_f1_means = []
    # mnb_auc_means = []

    # Container used to feed into charts
    mean_values = []

    # Run the model with final stats multiple times to get a mean of all
    for run in range(num_tests):
        # Create test train split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )

        # Instantiate a Multinomial Naive Bayes Model
        mnb_model = MultinomialNB(**best_params)

        # There scores
        # Fit the Model to the data
        mnb_model.fit(X_train, y_train)

        # find the best threshold
        threshold = thresh

        # calculate class labels using probabilities
        y_hat = mnb_model.predict_proba(X_test)[:, 1] >= threshold
        y_hat = np.where(y_hat, 1, 0)

        # Compute the metrics of the MultinomialNB model
        # fpr, tpr, thresholds = mt.roc_curve(y_test, y_hat_proba, pos_label=1)
        mnb_accuracy_means.append(round(accuracy_score(y_test, y_hat), 3))
        mnb_precision_means.append(round(precision_score(y_test, y_hat), 3))
        mnb_recall_means.append(round(recall_score(y_test, y_hat), 3))
        mnb_f1_means.append(round(mt.f1_score(y_test, y_hat), 3))

    mean_values.append(np.mean(mnb_accuracy_means))
    mean_values.append(np.mean(mnb_precision_means))
    mean_values.append(np.mean(mnb_recall_means))
    mean_values.append(np.mean(mnb_f1_means))

    # Create the barplot figure
    fig, ax = plt.subplots()
    bars = ax.barh(names, mean_values)
    ax.bar_label(bars)
    plt.yticks(names)
    plt.title(
        'Average Mean Scores. MultinomialNB |'
        ' params={} | ProbThreshold = {} |'
        ' Num_Runs={}'.format(str(best_params), threshold, num_tests))
    plt.xlabel('Percentage', fontsize=11, color='blue')
    plt.ylabel('Metrics', fontsize=11, color='blue')
    plt.show()

    # Create list of lists for the boxplot
    values = [mnb_accuracy_means, mnb_precision_means, mnb_recall_means,
              mnb_f1_means]

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


if __name__ == "__main__":
    # Grab arguments
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-d", "--directory", help="Path to Spam Directory", metavar='\b')
    args = parser.parse_args()

    # Raise an exception if directory path does not exist.
    if args.directory is not None:
        # Validate paths
        if os.path.exists(args.directory):
            dir_loc = args.directory
        else:
            raise Exception("Input files could not be found...")

    # Can't find input files. Raise exception
    else:
        input_err = (
            "Error: Spam Directory could not be found. Please specifiy"
            " location using --directory ")

        raise Exception(input_err)

    # Start timer
    start_time = time.perf_counter()

    # Create EmailProcessing instance
    email_obj = EmailProcessing()
    email_data_df = email_obj.process(args.directory, n_jobs=-1)

    # end timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print('Total time to generate DataFrame: {}'.format(
        round(elapsed_time, 2)))

    mnb_params = get_best_params(email_data_df)

    # Get the threshold
    results = []
    thrsh_runs = 5
    for i in range(thrsh_runs):
        mnb_threshold = get_best_threshold(
            email_data_df, mnb_params, show_chart=False
        )
        results.append(mnb_threshold)

    threshold_avg = sum(results) / len(results)
    print("average threshold:", threshold_avg)

    model_metrics = get_NB_results_and_plots(
        email_data_df, mnb_params, threshold_avg, 5
    )

    display_model_metrics(model_metrics)

    print("Complete...")
