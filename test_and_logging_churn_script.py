'''
Module testining customer churn analysis functions
Author : Nurudeen Adesina
Date : 23th August 2022
'''

import os
import logging
import churn_library as cls
import pytest
from math import ceil

# Categorical Features
cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="module")
def path():
    """
    Fixture - The test function test_import_data() will 
    use the return of path() as an argument
    """
    return "./data/bank_data.csv"

def test_import(path):
    '''
    test import_data function from the churn_library module
    '''
    # test file availaibilty in the path
    try:
        dataframe = cls.import_data(path)
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

def test_eda(path):
    '''
    Test perform_eda function in the churn_library module
    '''
    dataframe = cls.import_data(path)
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
        logging.error('No such file in folder')
        raise err

    try:
        assert os.path.isfile("./images/eda/customer_age_distribution.png") is True
        logging.info('File %s was found', 'customer_age_distribution.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err

    try:
        assert os.path.isfile("./images/eda/marital_status_distribution.png") is True
        logging.info('File %s was found', 'marital_status_distribution.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err

    try:
        assert os.path.isfile("./images/eda/total_transaction_distribution.png") is True
        logging.info('File %s was found', 'total_transaction_distribution.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
         
def test_encoder_helper(path):
    '''
    Test encoder_helper function in the churn_library module
    '''
    # Load DataFrame
    dataframe = cls.import_data(path)

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
        logging.info("Testing encoder_helper(data_frame, cat_columns=[]) - with empty cat_columns and None response: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper(data_frame, cat_columns=[] - with empty cat_columns and None response): ERROR")
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
            "Testing encoder_helper(dataframe, category_lst=cat_columns, response=None) - with non empty cat_columns and None response: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(dataframe, category_lst=cat_columns, response=None) - with non empty cat_columns and None response: ERROR")
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
        "Testing encoder_helper(dataframe, category_lst=cat_columns, response='Churn') - with non empty cat_columns and string response: SUCCESS")
    except AssertionError as err:
        logging.error(
        "Testing encoder_helper(dataframe, category_lst=cat_columns, response='Churn') - with non empty cat_columns and string response: ERROR")
        raise err

def test_perform_feature_engineering(path):
    '''
    Test perform_feature_engineering function in the churn_library module
    '''
    # Load the DataFrame
    dataframe = cls.import_data(path)

    #Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val=="Existing Customer" else 1)

    try:
        (_, X_test, _, _) = cls.perform_feature_engineering(      
                                                    dataframe=dataframe, 
                                                    category_lst=cat_columns,
                                                    response='Churn')

        # "Churn" should be present in dataframe's column name
        assert 'Churn' in dataframe.columns
        logging.info("Testing perform_feature_engineering. Column name(s) of the Dataframe contains the response string 'Churn': SUCCESS")
    except KeyError as err:
        logging.error("Column name(s) of the Dataframe does not contain the response string 'Churn': ERROR")
        raise err
        
    try:
        # X_test size should be 30% of `data_frame`
        assert (X_test.shape[0] == ceil(dataframe.shape[0]*0.3)) is True   
        logging.info(
            'Testing perform_feature_engineering. Test DataFrame size is correct: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering. Test DataFrame size is not correct: ERROR')
        raise err

def test_train_models(path):
    '''
    Test train_models() function from the churn_library module
    '''
    # Load the DataFrame
    dataframe = cls.import_data(path)

    # Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val=="Existing Customer" else 1)

    # Feature engineering 
    (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(  
                                                    dataframe=dataframe,
                                                    category_lst = cat_columns,
                                                    response='Churn')

    # Assert if 'logistic_model.pkl' file exist in results folder
    try:
        cls.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('File %s was found', 'logistic_model.pkl')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
        
    # Assert if 'rfc_model.pkl' file exist
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('File %s was found', 'rfc_model.pkl')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
        
    # Assert if 'roc_curve_result.png' file exist in results folder
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('File %s was found', 'roc_curve_result.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
        
    # Assert if 'rf_results.png' file exist in results folder
    try:
        assert os.path.isfile('./images/results/rf_result.png') is True
        logging.info('File %s was found', 'rf_result.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err

    # Assert if 'logistic_results.png' file exist in results folder
    try:
        assert os.path.isfile('./images/results/logistic_result.png') is True
        logging.info('File %s was found', 'logistic_result.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
    
   
    # Assert if 'feature_importances.png' file exists in results folder
    try:
        assert os.path.isfile('./images/results/feature_importances.png') is True
        logging.info('File %s was found', 'feature_importances.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
        
     # Assert if 'explainer.png' file exists in results folder
    try:
        assert os.path.isfile('./images/results/explainer.png') is True
        logging.info('File %s was found', 'explainer.png')
    except AssertionError as err:
        logging.error('No such file in folder')
        raise err
        
if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    cat_columns = [ 'Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category'  ]
    test_perform_feature_engineering()
    test_train_models()
    