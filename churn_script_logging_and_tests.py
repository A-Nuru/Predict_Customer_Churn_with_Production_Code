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

def test_import():
    '''
    test import_data function from the churn_library module
    '''
    # test file availaibilty in the path
    try:
        dataframe = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    
    # test the the dataframe
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info("Testing import_data: The file has %d rows and %d columns",
                     dataframe.shape[0], dataframe.shape[1])
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda():
    '''
    Test perform_eda function in the churn_library module
    '''
    dataframe = cls.import_data("./data/bank_data.csv")
    try:
        cls.perform_eda(dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error('Column %s not found', err.args[0])
        #raise err
        
     # Assert that the plots are created and saved
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        logging.info('File %s was found', 'churn_distribution.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile("./images/eda/customer_age_distribution.png") is True
        logging.info('File %s was found', 'customer_age_distribution.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile("./images/eda/marital_status_distribution.png") is True
        logging.info('File %s was found', 'marital_status_distribution.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile("./images/eda/total_transaction_distribution.png") is True
        logging.info('File %s was found', 'total_transaction_distribution.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
         
def test_encoder_helper():
    '''
    Test encoder_helper() function from the churn_library module
    '''
    # Load DataFrame
    dataframe = cls.import_data("./data/bank_data.csv")

    # Create `Churn` feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
                                apply(lambda val: 0 if val=="Existing Customer" else 1)

    # Categorical Features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    try:
        df_encoded = cls.encoder_helper(
                            dataframe=dataframe,
                            category_lst=[],
                            response=None)

        # Data should be the same
        assert df_encoded.equals(dataframe) is True
        logging.info("Testing encoder_helper(data_frame, category_lst=[]): SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper(data_frame, category_lst=[]): ERROR")
        raise err

    try:
        df_encoded = cls.encoder_helper(
                            dataframe=dataframe,
                            category_lst=cat_columns,
                            response=None)

        # Column names should be same 
        assert df_encoded.columns.equals(dataframe.columns) is True

        # Data should be different
        assert df_encoded.equals(dataframe) is False
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): ERROR")
        raise err

    try:
        df_encoded = cls.encoder_helper(
                            dataframe=dataframe,
                            category_lst=cat_columns,
                            response='Churn')

        # Columns names should be different
        assert df_encoded.columns.equals(dataframe.columns) is False   

        # Data should be different
        assert df_encoded.equals(dataframe) is False

        # Number of columns in encoded_df is the sum of columns in data_frame and the newly created columns from cat_columns
        assert len(df_encoded.columns) == len(dataframe.columns) + len(cat_columns)    
        logging.info(
        "Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): SUCCESS")
    except AssertionError as err:
        logging.error(
        "Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): ERROR")
        raise err

if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    