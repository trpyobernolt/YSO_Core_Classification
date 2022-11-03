#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:26:12 2022

@author: troyobernolte

Testins Machine Learning Algorithms for conglomerated Data
"""

import data_reader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn import preprocessing


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
pd.options.mode.chained_assignment = None

# LOAD DATA 

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
        return RandomForestClassifier(max_depth=3, n_estimators=21, max_features="auto")
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
        self.X, self.Y = data_reader.main()
        self.Y = self.Y.astype('int')
        
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

        for col in cat_col_sel(self.X):
            self.X[col].str.strip()
            self.X[col] = self.X[col].replace('-', np.nan, regex=True)
            self.X[col] = self.X[col].replace(r'^\s*$', np.nan, regex=True)
               
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

model1 = Model("Random Forest Classifier", [preprocessing.MaxAbsScaler()])
model1.data_engineer()
model1.scale_data()
model1.split_data(0.3)
model1.run()


