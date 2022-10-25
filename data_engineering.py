#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:24:26 2022

@author: troyobernolte

This file is meant to engineer the data both to be readable and to trim the data
for better results in learning models

REQUIRES file from running data_reader
"""

import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
import numpy as np

X = dataset = pd.read_csv('Data/X_vals.csv')
Y = dataset = pd.read_csv('Data/Y_vals.csv')


def data_modification():
    """Runs data modification and engineering functions"""
    return

def engineer_data():
    #TO DO
    return

def fill_data():
    """Takes in data and fills any values that are undefined"""
    global X
    cat_col_sel = selector(dtype_include = object)
    for col in cat_col_sel(X):
       X[col] = X[col].str.strip()
       X[col] = X[col].replace(r'^\s*$', np.nan, regex=True)
       for i in range(len(X[col])):
           if X.loc[i][col] == '153749.6-331545':
               X.loc[i][col] = float(-177795.4)
               print("THIS ONE")
    imputer_x = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imputer_x.fit(X)
    X = imputer_x.transform(X)
    return

fill_data()

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=3, n_estimators=21, max_features=9)
classifier.fit(X, Y) 