#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:23:48 2023

@author: Ryan Herrin
"""
import os
import csv
import time
import warnings
import itertools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate


program_desc = """
A class that takes in classifier along with hyperparameters and will run every
combination of each classifier and parameter. Will return in object with all
results as well as output charts of model performances into the directory this
script is ran from.
"""

# Matplotlib properties
matplotlib.use('Agg')
matplotlib.style.use('default')


# Warning supression
def warn(*args, **kwargs):
    '''Supress the sklearn warnings'''
    pass


warnings.warn = warn


class BFMSearch:
    '''Python Scripts that simulates a gridsearch-like function that takes in
    a classifier with hyperparameters and generates different results based on
    the combinations of hyperparameters'''

    def __init__(self):
        self.classifier = None
        self.models = dict()
        self.num_of_folds = 5  # Default is 5
        self.verbose = False
        self.data = None
        self.output_dir = os.getcwd() + '/output'
        self.target = None
        self.score = None
        self._outdir_ready = None
        self.results = None

    def _log(self, x):
        '''Simple logger for formatting output'''
        if self.verbose:
            print('[BFMSearch] > {}'.format(str(x)))
        else:
            pass

    def _get_data(self, data):
        '''Determines the data to be used if data was passed or not'''
        if data is not None:
            self.data = data
        else:
            # Global data used for demonstration
            raise Exception('Data was not provided')

        return self.data

    def _set_metric_classification(self, metric='macro'):
        """Change the metric scores based on if the target data is binary
        or not."""
        if metric == 'binary':
            self.score = ['accuracy', 'precision', 'recall', 'f1']

        if metric == 'weighted':
            self.score = ['accuracy', 'precision_weighted', 'recall_weighted',
                          'f1_weighted']

        if metric == 'macro':
            self.score = ['accuracy', 'precision_macro', 'recall_macro',
                          'f1_macro']

        if metric == 'micro':
            self.score = ['accuracy', 'precision_micro', 'recall_micro',
                          'f1_macro']

    def _create_configurations(self, clsf_arg):
        '''Generate a dictionary of all available CLF combinations to run'''
        # Indexor for appending to configuration dict
        dict_i = 0

        # If clsf_arg is a list then it's as simple as appending
        if isinstance(clsf_arg, list):
            for clsf in clsf_arg:
                self.models[dict_i] = dict()
                dict_i += 1

        elif isinstance(clsf_arg, dict):
            # Loop through the classifiers
            for clf in clsf_arg:
                # Now go through all the params and create the different
                # combinations for them
                param_vals = []
                name_of_params = list(clsf_arg[clf].keys())

                # For the number of parameters
                for param in clsf_arg[clf]:
                    param_vals.append(clsf_arg[clf][param])

                # Create combinations
                param_combs = list(itertools.product(*param_vals))

                # Add each combination to the models dictionary
                for p_c in param_combs:
                    self.models[dict_i] = {clf: dict()}
                    for p in range(len(name_of_params)):
                        self.models[
                            dict_i][clf][name_of_params[p]] = p_c[p]

                    # Add the calable model with the params to the model
                    self.models[dict_i]['model'] = clf(
                        **self.models[dict_i][clf])

                    dict_i += 1

        else:
            raise TypeError("a_clf must be of type list or dict...")

    def _run_model_fits(self):
        '''Iterate through the models dict and fit the model to the data. Adds
        a value to the models dictionary indicating if the model succeeded in
        fitting the data or not.'''
        modl_cnt = 1  # For keeping track of the number of models fitted
        for mdl in self.models:
            # Verbose option for keeping track of model fit progress
            if self.verbose:
                print('[BFMSearch] > fitting model {} out of {}'.format(
                    modl_cnt, len(self.models)))

            try:
                self.models[mdl]['model'].fit(self.data, self.target)
                self.models[mdl]['is_fit'] = True

            except Exception as err:
                self._log(str(err))
                # If False, we will prevent this model from running in future
                # processes
                self.models[mdl]['is_fit'] = False

            modl_cnt += 1

        # Insert a space from the model fit loading status
        if self.verbose:
            print('')

    def _generate_results(self):
        '''Generate the results by prediction and retriveing the Accuracy and
        other metrics.'''
        res_count = 1
        # Loop through all of the models
        for mdl in self.models:
            # Verbose option for keeping track of model results progress
            if self.verbose:
                print('[BFMSearch] > Generating results for model {} '
                      'out of {}'.format(
                          res_count, len(self.models)))

            # If the model was able to fit the data then lets run it
            if self.models[mdl]['is_fit']:
                # Create KFold train, test data
                kf = KFold(n_splits=self.n_folds)
                # Run the Cross validation model
                self.models[mdl]['scores'] = cross_validate(
                    self.models[mdl]['model'],
                    self.data,
                    self.target,
                    scoring=self.score,
                    cv=kf,
                    n_jobs=-1
                )

                # Output is a dictionary with these values that we need
                # 'test_accuracy', 'test_precision', 'test_recall', 'test_f1'

                # Get the average Accuracy score
                try:
                    self.models[mdl]['avg_acc'] = (
                        sum(
                            self.models[mdl]['scores']['test_accuracy']
                        ) / self.n_folds
                    )
                except Exception as err:
                    err = err
                    print("Failed to get Accuracy")
                    self.models[mdl]['avg_acc'] = 'NA'

                # Get average precision score
                try:
                    self.models[mdl]['avg_prec'] = (
                        sum(
                            self.models[mdl]['scores']['test_' + self.score[1]]
                        ) / self.n_folds
                    )
                except Exception as err:
                    err = err
                    print("Failed to get Precision")
                    self.models[mdl]['avg_prec'] = 'NA'

                # Get average recall score
                try:
                    self.models[mdl]['avg_recall'] = (
                        sum(
                            self.models[mdl]['scores']['test_' + self.score[2]]
                        ) / self.n_folds
                    )
                except Exception as err:
                    err = err
                    print("Failed to get Recall")
                    self.models[mdl]['avg_recall'] = 'NA'

                # Get average precision score
                try:
                    self.models[mdl]['avg_f1'] = (
                        sum(
                            self.models[mdl]['scores']['test_' + self.score[3]]
                        ) / self.n_folds
                    )
                except Exception as err:
                    err = err
                    print("Failed to get F1")
                    self.models[mdl]['avg_f1'] = 'NA'

            # Increment the count for models done regenerating results.
            res_count += 1

        # Break from the carrage return
        if self.verbose:
            print('')

    def _get_output_dir(self):
        '''Check for output directory and create sub directory specific for
        each run'''
        # Assign unique name to output run
        dir_timestamp = str(time.time()).split('.')[0]

        # Check to see if the output file exists and create one if it doesn't
        if not os.path.exists(self.output_dir):
            try:
                os.mkdir(self.output_dir)

            except Exception as err:
                self._log(str(err))
                self._outdir_ready = False

                return None

        # Create the outfile for the unique run
        self.output_dir = self.output_dir + '/{}'.format(dir_timestamp)
        try:
            os.mkdir(self.output_dir)
            self._outdir_ready = True

        except Exception as err:
            self._log(str(err))
            self._outdir_ready = False

    def _create_summary_report(self):
        '''creates a csv file of the summary of models'''
        csv_file_name = self.output_dir + '/Results_Summary.csv'
        csv_header = ['Run', 'Classifier', 'Parameters', 'is_fit', 'Acc',
                      'Precision', 'Recall', 'F1']

        # Create lists to better organize when writing to a csv
        results = []  # Master list

        # Counter to corresponding models runs with results on charts
        run_count = 1

        # Loop though the models dict and append the results to the lists
        for mdl in self.models:
            _mdl_rslt = []
            _mdl_rslt.append(run_count)
            run_count += 1

            # Extract to classifier
            for key in self.models[mdl].keys():
                _non_clf_keys = ['model', 'is_fit', 'test_accuracy', 'scores',
                                 'avg_acc', 'avg_prec', 'avg_recall', 'avg_f1']
                _non_clf_keys.extend(self.score)

                if key not in _non_clf_keys:
                    _mdl_rslt.append(str(key).split('.')[-1].split("'")[0])
                    # Get the parameters
                    _mdl_rslt.append(str(self.models[mdl][key]))

            # Get the is_fit value
            _mdl_rslt.append(str(self.models[mdl]['is_fit']))

            if self.models[mdl]['is_fit']:
                # Get the Accuracy
                _mdl_rslt.append(self.models[mdl]['avg_acc'])
                # Get the Precision
                _mdl_rslt.append(self.models[mdl]['avg_prec'])
                # Get the Recall
                _mdl_rslt.append(self.models[mdl]['avg_recall'])
                # Get the F1
                _mdl_rslt.append(self.models[mdl]['avg_f1'])
            else:
                _mdl_rslt.append('na')
                _mdl_rslt.append('na')
                _mdl_rslt.append('na')
                _mdl_rslt.append('na')

            # Append result set to master results list
            results.append(_mdl_rslt)

        self.results = results

        # Write out to the csv file
        with open(csv_file_name, 'w', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(csv_header)
            csv_writer.writerows(results)

    def _generate_boxplot(self):
        '''Generates a box plot of all runs'''
        # Reuse the Results list since we already have the stats from that
        names = []
        values = []
        for res in range(len(self.results)):
            names.append(self.results[res][0])
            values.append(self.results[res][5])

        # Create the boxplot figure
        bx_plt = plt.figure()
        bx_plt.suptitle('Model Score Variance Comparison')
        ax = bx_plt.add_subplot(111)
        plt.boxplot(values)
        plt.grid()
        plt.tight_layout(pad=1.5)
        ax.set_xticklabels(names)
        plt.xlabel('Run # - (Maps to each run in the Results_Summary.csv)',
                   fontsize=11, color='blue')
        plt.ylabel('Score: {}'.format(self.score), fontsize=11, color='blue')

        # Save the figure
        plt.savefig(self.output_dir + '/BoxPlot_Compare.pdf')

    def _generate_barplot(self):
        names = []
        values = []
        for res in range(len(self.results)):
            names.append(self.results[res][0])
            values.append(round(self.results[res][4], 3))

        # Create the barplot figure
        fig, ax = plt.subplots()
        bars = ax.barh(names, values)
        ax.bar_label(bars)
        plt.yticks(names)
        plt.title('Model Score Comparison')
        plt.ylabel('Run # - (Maps to each run in the Results_Summary.csv)',
                   fontsize=11, color='blue')
        plt.xlabel('Score: {}'.format(self.score), fontsize=11, color='blue')

        # Save the figure
        plt.savefig(self.output_dir + '/Bar_Score_Compare.pdf')

    def display_top_models(self, on_metric="accuracy", n=10):
        """Print out the top models and their parameters based on the chosen
        metric.

        Parameters
        ----------
        on_metric: str ["accuracy", "preceision", "recall", "f1"]
            What metric to base the top scores on
        n: int
            How many models to print out

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def get_best_model(self, on_metric="accuracy"):
        """
        Parameters
        ----------
        on_metric : str, optional
            What metric to get the best model on. The default is "accuracy".

        Returns
        -------
        None.
        """
        raise NotImplementedError()

    def create_charts(self):
        """Create output Bar and Boxplots charts. Max of 10 models will be
        displayed"""
        raise NotImplementedError()

        self._generate_boxplot()
        self._generate_barplot()

    def run(self, a_clf, data, target, n_folds=5, verbose=True,
            metric='macro', create_csv=False):
        '''Runs and stores the results of the classifier passed through with
        all combinations of hyperparameters if provided.

        Parameters
        ----------
        a_clf : list or dict
            A list of classifiers or a dict of classifiers with hyperparemeters
            as nested dictionaries with a list of values. (See Examples..)
        data : array-like objects
            Data to use for classifiers. If left as default the script will use
            the sample imported data
        target : array-like object
            Target column of the data for supervised learning
        n_folds : int Default = 5
            Number of cross validation folds.
        verbose : bool
            For those who just like to see output on the console
        metric : ['binary', 'weighted', 'macro', 'micro'] Default='macro'
            How to measure fit. Default = 'accuracy'
        create_csv : bool
            Generate a CSV output file of the model results. Default is False

        Returns
        -------
        None

        Examples
        --------
        # a_clf if passed as a dict with hyperparams
        a_clf_ex = {
            Clsf_1: {'H_Param': [<values>], 'H_Param': [<values>]},
            Clsf_2: {'H_Param': [<values>], 'H_Param': [<values>]}
        }

        # or a_clf as just a list of classifiers
        a_clf_ex = [clf_1, clf_2, clf_3]

        obj = DeathToGS()
        obj.run()
        '''
        # Set default values
        self.verbose = verbose
        self.n_folds = n_folds
        self._set_metric_classification(metric)

        # determine which data set to use
        data = self._get_data(data)
        self.target = target

        # Get the configurations
        self._log("Creating model configurations...")
        self._create_configurations(a_clf)

        # Fit all the models to the data
        self._log("Fitting models to data...")
        self._run_model_fits()

        # Generate predicitons and metrics
        self._log('Running Models...')
        self._generate_results()

        if create_csv:
            # Generate outputs for the results
            self._get_output_dir()

            # If the output directory is craeted then continue with generating
            self._log('Generating Outputs...')

            if self._outdir_ready:
                self._create_summary_report()

            self._log("Complete. Results are stored at {}".format(
                self.output_dir))

        else:
            self._log("Complete")

