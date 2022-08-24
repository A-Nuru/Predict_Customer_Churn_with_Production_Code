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
    
if __name__ == '__main__':
    DF = import_data(pth='./data/bank_data.csv')
    perform_eda(DF)    
