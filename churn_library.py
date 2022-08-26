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

def perform_feature_engineering(dataframe, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [ 'Gender', 'Education_Level', 'Marital_Status','Income_Category', 'Card_Category'  ]

    # feature engineering
    df_encoded = encoder_helper(dataframe=dataframe, category_lst=cat_columns, response=response)

    # target feature 
    y = df_encoded['Churn']     

    # Create dataframe
    X = pd.DataFrame()         

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    # Features DataFrame
    X[keep_cols] = df_encoded[keep_cols]

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
    #print(X_train)
    return (X_train, X_test, y_train, y_test)

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Plot RandomForestClassifier classification report
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, 
             fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, 
             fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, 
             fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, 
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')
    
    # Plot LogisticRegression classification report
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass
    
if __name__ == '__main__':
    DF = import_data(pth='./data/bank_data.csv')
    DATAFRAME = perform_eda(DF)
    #category_lst = category_lst(DF)
    cat_columns = ['Gender',
 'Education_Level',
 'Marital_Status',
 'Income_Category',
 'Card_Category']
    DF_ENCODED = encoder_helper(DATAFRAME, cat_columns, 'Churn')
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DF_ENCODED, response='Churn')
