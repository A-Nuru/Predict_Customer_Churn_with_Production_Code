'''
Module testining customer churn analysis functions
Author : Nurudeen Adesina
Date : 23th August 2022
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test import_data function from the churn_library module
    '''
    # test file availaibilty in the path
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    
    # test the the datafram
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: The file has {} rows and {} columns".format(df.shape[0], df.shape[1]))
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda():
    '''
    Test perform_eda function in the churn_library module
    '''
    df = clib.import_data("./data/bank_data.csv")
    try:
        cls.perform_eda(dataframe=dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error('Column {} not found'.format(err.args[0]))
        raise err