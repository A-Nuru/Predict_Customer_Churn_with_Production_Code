# Predict Customer Churn

- Project **Predict Customer Churn**

## Project Description
The goal of this project is to identify the bank credit card users most likely churn. The Project 
will include a Python package for a machine learning that predicts whether the credit card users 
of a bank will churn or not. Best software and engineering practices for PEP8 such as clean 
and modular code, documentation, logging and test driven development will be employed to  produce 
a production code.It involves
- Loading data
- Preparing the data
- Exploring the data to understand it
- Feature engineering
- Building model
- Evaluating model
- Storing models, images, results and logs

## Files and data description
Overview of the files and data present in the root directory. 
- churn_notebook.ipynb - ipython notebook
- churn_library.py - python module
- churn_script_logging_and_tests.py - test file for the churn_library  module
- requirements_py3.8.txt - Contains the project installation prerequisites
- Readme.md - a readme project file 
- license.txt - a GNU license file
- data folder - contains the bank_data.csv
- image folder - Contains the eda and results subfolder. The eda subfolder stores the saved plots.
                 The results subfolder stores the roc curve, logistic regression, random forest and feature importances png results 
- logs folder - Contains the churn_library log file
- models folder - Contains the stored models for easy reuse

- data folder - Contains the bank_data.csv file. The data used in this project are the credit card data of a bank. The data is obtained from kaggle and can be viewed here

## Running Files
How do you run your files? What should happen when you run your files?
- Create a virtual OR conda environment and activate
- Install dependencies
`python -m pip install -r requirements_py3.8.txt`
- Running the churn_library.py 
`ipython churn_library.py`  OR  `python churn_library.py`

- Running the churn_script_logging_and_tests.py
`ipython churn_script_logging_and_tests.py` OR `python churn_script_logging_and_tests.py`
- Style Guide. Run
`autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py`
`churn_script_logging_and_tests.py`
`autopep8 --in-place --aggressive --aggressive churn_library.py`

- Style Checking and Error Spotting. Run
`pylint churn_library.py`
`pylint churn_script_logging_and_tests.py`


