'''
Module containing customer churn analysis functions
Author : Nurudeen Adesina
Date : 23th August 2022
'''

# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve
os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(pth):
    '''
    Returns a pandas dataframe for the csv found at 'pth'

    input:
            pth: a path to the csv file
    output:
            dataframe: pandas dataframe
    '''
    return pd.read_csv(pth)

def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    dataframe.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Encoding the label column
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    # plotting and saving figures
    # Churn Distribution
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'],kde=True)
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')
    return dataframe

def category_lst(dataframe):
    cat_columns = []
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            cat_columns.append(column)
    return cat_columns.remove('Attrition_Flag')

def quantitative_columns(df):
    quant_columns = []
    for column in df.columns:
        if df[column].dtype == 'int64' or df[column].dtype == 'float64':
            quant_columns.append(column)
    return quant_columns
#category_lst = category_lst(DF)

def encoder_helper(dataframe, category_lst, response = None):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the churn notebook
    input:
            data_frame: pandas DataFrame
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
                      naming variables or index y column]
    output:
            data_frame: pandas DataFrame with new columns for analysis
    '''
    # Copy DataFrame
    df_encoded = dataframe.copy(deep=True)
    
    for category in category_lst:
        column_lst = []
        column_groups = dataframe.groupby(category).mean()['Churn']

        for val in dataframe[category]:
            column_lst.append(column_groups.loc[val])

        if response:
            df_encoded[category + '_' + response] = column_lst
        else:
            df_encoded[category] = column_lst
    print(df_encoded)
    return df_encoded
      
if __name__ == '__main__':
    DF = import_data(pth='./data/bank_data.csv')
    DATAFRAME = perform_eda(DF)
    #category_lst = category_lst(DF)
    category_lst = ['Gender',
 'Education_Level',
 'Marital_Status',
 'Income_Category',
 'Card_Category']
    DF_ENCODED = encoder_helper(DATAFRAME, category_lst, 'churn')
