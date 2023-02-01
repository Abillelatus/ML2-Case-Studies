#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 08:49:00 2023

@author: xclusive
"""

import os
import time
import email
import argparse
from io import StringIO
import multiprocessing as m_proc
from multiprocessing import Pool
from html.parser import HTMLParser


# Hardcoded path for testing
dflt_path = ('/home/xclusive/rTek/School/SMU/MSDS_7333_Quantifying_The_World/'
             'QTW-Case-Studies/Datasets/SpamMessages')

# Description for program
program_desc = '''
'''


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

    def _build_file_path_list(self):
        '''Walk through the root folder and build a list of absolute paths to
        the emails'''
        ignore_suffix = ['.py', '.ipynb', '.tmp']  # Files to ignore
        for root, dirs, files in os.walk(self.path):
            for name in files:
                # Make sure we are not bringing in extra files
                if any(ext not in ignore_suffix for ext in os.path.join(
                        root, name)):
                    self.file_paths.append(os.path.join(root, name))
                else:
                    pass

    def _process_html(self, html_file):
        '''If HTML file, strip tags and return data'''
        strip_fx = self.MLStripper()
        strip_fx.feed(html_file)

        return strip_fx.get_data()

    def _read_email_contents(self, file_pool):
        '''Attempt to read the contents of an email and store the data as a
        dictionary where:
            {
                str(filename) : [int(is_spam), str(payload)
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

                        # Add to the dictionary
                        email_content[file_name] = [is_spam, data]

        except Exception as err:
            print(str(err))

        return email_content

    def _process_payload(self, payload_data):
        '''Process the payload before passing it back.'''
        # If the data is html, then strip the html tags
        _data = payload_data

        # TODO: Need to find a way to handle payloads that are cast as lists
        if isinstance(_data, list):
            return _data
        else:
            # If it's HTML data, strip the tags and return only the data
            if '<html>' in _data.lower():
                _data = self._process_html(_data)

        return _data

    def _create_word_symbol_list(self, payload_data):
        '''Parse through the data and create an word and symbol dictionary'''
        return payload_data

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
        '''Function Execution Director for multiprocssing'''
        # Process the emails
        email_contents = self._read_email_contents(file_list)
        # Create a word list from the payload data
        email_data = self._create_word_symbol_list(email_contents)

        return email_data

    def process(self, path_to_email_dirs=None):
        '''Iterate through the emails process the information.
        Returns:
        pandas DataFrame
        '''
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

        # Run multiprocessing using the Pool API
        with Pool(self.cpu_core_count) as p:
            pool_results = p.map(self._mp_executor, process_pool)

        return pool_results


if __name__ == "__main__":
    # Grab arguments
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(
        "-d", "--directory", help="Path to diabetic_data.csv", metavar='\b')
    args = parser.parse_args()

    # If arguments are provided then treat them as the defualt location for
    # input files. Other-wise we will check if the files are in the default
    # location. Raise an exception if no files are found.
    if args.directory is not None:
        # Validate paths
        if os.path.exists(args.directory):
            dir_loc = args.directory
        else:
            raise Exception("Input files could not be found...")

    # If no arguments are provided, see if file is in default location
    elif args.directory is None:
        if os.path.exists(dflt_path):
            dir_loc = dflt_path
        else:
            raise Exception("Error: diabetic_data.csv could not be found.",
                            "Please specifiy location using --directory ")

    # Can't find input files. Raise exception
    else:
        input_err = (
            "Error: diabetic_data.csv could not be found. Please specifiy",
            "location using --directory ")

        raise Exception(input_err)

    start_time = time.perf_counter()

    # Create EmailProcessing instance
    email_info = EmailProcessing()
    data = email_info.process(dflt_path)

    # end timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(elapsed_time)

