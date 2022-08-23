'''
Module containing customer churn analysis functions
Author : Nurudeen Adesina
Date : 23th August 2022
'''

# import libraries
import os
#import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
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
            df: pandas dataframe
    '''	
    return pd.read_csv(pth)

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Encoding the label column
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    