#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:46:52 2022

@author: troyobernolte
"""

"""This module is meant to get statistics for the given data."""
import pandas as pd
from midterm_model import Model
from matplotlib import pyplot as plt
from sklearn.compose import make_column_selector as selector
import numpy as np


dataset = pd.read_csv('corona_australia_table1.tsv', delimiter=';',comment='#')
labels = ['Signi070', 'Sp070', 'e_Sp070',
             'Sp/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070', 'FWHMb070',
               'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp/Sbg160', 'Sconv160',
               'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160', 'Signi250',
               'Sp250', 'e_Sp250', 'Sp/Sbg250', 'Sconv250', 'Stot250', 'e_Stot250',
               'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350', 'e_Sp350',
               'Sp/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350', 'FWHMa350', 'FWHMb350',
               'PA350', 'Signi500', 'Sp500', 'e_Sp500', 'Sp/Sbg500', 'Stot500',
               'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500', 'SigniNH2', 'NpH2',
               'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2', 'FWHMbNH2', 'PANH2', 'NSED',
               'CSARflag', 'CUTEXflag']
X = dataset[labels]
Y = dataset['Coretype']

def shapes(model):
        #Get dataframes
        X_df = pd.DataFrame(model.X)
        Y_df = pd.DataFrame(model.Y)
        x_train_df = pd.DataFrame(model.x_train)
        x_test_df = pd.DataFrame(model.x_test)
        y_train_df = pd.DataFrame(model.y_train)
        y_test_df = pd.DataFrame(model.y_test)
        
        #print shapes
        print("\nShape of X data is: ", X_df.shape)
        print("Shape of Y data is: ", Y_df.shape)
        print("Shape of X training data is: ", x_train_df.shape)
        print("Shape of X testing data is: ", x_test_df.shape)
        print("Shape of Y training data is: ", y_train_df.shape)
        print("Shape of Y testing data is: ", y_test_df.shape)

def describe(model):
        X_df = pd.DataFrame(model.X)
        #Print descriptions
        print("Description of X Data: ")
        print(X_df.describe())

def summary_stats():
    stats = []
    #Create dummy model
    model = Model("KNN", [])
    model.split_data(0.3)
    for i in range(len(labels)):
        column = model.X[i]
        data = []
        for point in range(len(column)):
            value = model.X[i][point]
            data.append(float(value))
        data_arr = np.asarray(data)
        stdev = np.std(data_arr)
        mean = np.mean(data_arr)
        minimum = np.amin(data_arr)
        maximum = np.max(data_arr)
        rng = maximum - minimum
        stats.append([labels[i], [stdev, mean, rng, minimum, maximum]])
    print("\n\nSummary Statistics")
    print("---------------------")
    for i in range(len(stats)):
        print(labels[i], "Has: ")
        print("\tStandard Deviation of", stats[i][1][0])
        print("\tMean of", stats[i][1][1])
        print("\tRange of", stats[i][1][2])
        print("\tMinimum of", stats[i][1][3])
        print("\tMaximum of", stats[i][1][4])

        
        

