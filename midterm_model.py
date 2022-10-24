#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:49:45 2022

@author: troyobernolte
"""

"""This module is meant to give the model function and class"""

# Import packages for data
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import Perceptron
from sklearn import tree

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import numpy as np
pd.options.mode.chained_assignment = None

# LOAD DATA 
# Convert dataset to a pandas dataframe:
dataset = pd.read_csv('corona_australia_table1.tsv', delimiter=';',comment='#')

#Defining  funciton to determine the model
def check_model(model):
    """Takes in a string representing the model and returns a classifier object of said model"""
    if(model == 'KNN'):
        return KNeighborsClassifier(n_neighbors=4)
    elif(model == 'Support Vector Machine'):
        return SVC(kernel="linear", C=0.025)
    elif(model == 'Gaussian Process Classifier'):
        return GaussianProcessClassifier(1.0 * RBF(1.0))
    elif(model == 'Random Forest Classifier'):
        return RandomForestClassifier(max_depth=3, n_estimators=21, max_features=9)
    elif(model == 'MLP Classifier'):
        return  MLPClassifier(alpha=2, max_iter=1000)
    elif(model == 'Ada Boost Classifier'):
        return AdaBoostClassifier(n_estimators=300, learning_rate=1.5)
    elif(model == 'GnB'):
        return GaussianNB()
    elif(model == "Perceptron"):
        return Perceptron(tol = 0.001, random_state = 0)
    elif(model == "Decision Tree"):
        return tree.DecisionTreeClassifier()

#Defining the class model
class Model:
    """A class with a user-defined machine learning model
    takes in a model defined as a string
    Can take in a list of preprocessing elements to scale data
    Can take in more args depending on the requested model for optimization"""
    
    def __init__(self, *args):
        self.X = dataset[['Signi070', 'Sp070', 'e_Sp070',
               'Sp/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070', 'FWHMb070',
               'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp/Sbg160', 'Sconv160',
               'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160', 'Signi250',
               'Sp250', 'e_Sp250', 'Sp/Sbg250', 'Sconv250', 'Stot250', 'e_Stot250',
               'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350', 'e_Sp350',
               'Sp/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350', 'FWHMa350', 'FWHMb350',
               'PA350', 'Signi500', 'Sp500', 'e_Sp500', 'Sp/Sbg500', 'Stot500',
               'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500', 'SigniNH2', 'NpH2',
               'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2', 'FWHMbNH2', 'PANH2', 'NSED',
               'CSARflag', 'CUTEXflag']]
        
        self.Y = dataset['Coretype']
        
        #Assign the approriate classifier (dynamic for optimization)
        if len(args) == 5:
            self.classifier = RandomForestClassifier(max_depth=args[2], 
                                                     n_estimators=args[3], 
                                                     max_features=args[4])
        else:
            self.classifier = check_model(args[0])
            
        #Label what kind of model is being used and assign scalars
        self.label = args[0]
        if len(args) > 1:
            self.scalars = args[1]
        
        #placeholders to make other methods work
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def data_engineer(self):
        #Strip file of any spaces
        cat_col_sel = selector(dtype_include = object)
        #Rows 1 and 2 don't contain any data
        self.X = self.X.drop(0)
        self.X = self.X.drop(1)
        
        #Dropping Signi070. Doesnt contain information and improves model
        self.X = self.X.drop('Signi070', axis = 1)
        self.X = self.X.drop('FWHMb350', axis = 1)
        self.X = self.X.drop('FWHMb500', axis = 1)
        #####self.X.FWHMbNH2 = pd.to
        self.X[self.X.FWHMbNH2 < np.percentile(self.X.FWHMbNH2, 99)]


        
        for col in cat_col_sel(self.X):
            self.X[col].str.strip()
            self.X[col] = self.X[col].replace('-', np.nan, regex=True)
            self.X[col] = self.X[col].replace(r'^\s*$', np.nan, regex=True)
            
        #Rows 1 and 2 don't contain any data
        self.Y = self.Y.drop(0)
        self.Y = self.Y.drop(1)
        self.Y.str.strip()
    
        #Inputting missing data.
        imputer_x = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        imputer_x.fit(self.X)
        self.X = imputer_x.transform(self.X)
        print(self.X)
        
        
    def split_data(self, sample_size):
        
        #Split the data into training sets and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = sample_size, random_state=42)



        
    def scale_data(self):
        """Scaling the X data with a given set of preprocessing args"""
        #Case for scaling data set after split
        if not(self.x_train == None and self.y_train == None):
            for scaler in self.scalars:
                scaler.fit(self.X)
                self.x_train = scaler.transform(self.x_train)
                self.x_test = scaler.transform(self.x_test)
                self.X = scaler.transform(self.X)
        #Case for scaling data before split
        else:
            for scaler in self.scalars:
                scaler.fit(self.X)
                self.X = scaler.transform(self.X)
        X_df = pd.DataFrame(self.X)
        print("\n Summary Statistics of the X dataframe after, ", scaler,
                  "\n", X_df.describe())
        
    
    def run(self):
        """Trains and runs the model. Returns the accuracy score"""
        #TRAIN MODEL
        self.classifier.fit(self.x_train, self.y_train) 
        
        # Predict y data with classifier: 
        self.y_predict = self.classifier.predict(self.x_test)
        print(confusion_matrix(self.y_test, self.y_predict))
        print("\n", classification_report(self.y_test, self.y_predict)) 
        print("\nTesting Set: ", self.y_test.array)
        print("\nPredicted Values", self.y_predict)
        return accuracy_score(self.y_test, self.y_predict)
    
    