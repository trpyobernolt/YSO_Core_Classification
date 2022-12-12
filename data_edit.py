#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:34:15 2022

@author: troyobernolte

This module takes data from data_reader, analyzes it, edits it, and cleans it
"""

import data_reader
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, \
QuantileTransformer
from sklearn import preprocessing, model_selection, feature_selection, \
ensemble, linear_model, metrics, decomposition, svm, naive_bayes, discriminant_analysis, \
kernel_ridge, neighbors, gaussian_process, tree



data_test = data_reader.main()

"""By running df.info(), we can see that all objects are categorized as objects
We want to categorize our floats as floats and ints as ints to analyze the data"""

"""Need to input data for Signi70
df["Signi70"] = df["Signi70"].str.strip()
df["Signi70"] = df["Signi70"].replace(r'^\s*$', np.nan, regex=True)"""

class Data_Cleaner:
    scalerTypes = {'standard':preprocessing.StandardScaler(), 'minmax':preprocessing.MinMaxScaler(), \
                   'robust':preprocessing.RobustScaler(with_centering = True, unit_variance = True), \
                   'quantileTransform':preprocessing.QuantileTransformer()}
        
    def __init__(self, scalers):
        self.df = data_reader.main()
        self.data_clean()
        self.data_fill()
        self.X = self.df.drop(['Coretype', 'Region'], axis=1)
        self.Y = self.df['Coretype']
        self.scale(scalers)
        
    def data_clean(self):
        """Cleans data. Should be run before anything else"""
        for col in self.df.select_dtypes(include='object').columns:
            for row in range(self.df.shape[0]):
                if not(col == "Region" or col == "Coretpye"):
                    try:
                        self.df[col][row] = self.df[col][row].str.strip()
                    except AttributeError:
                        try: 
                            self.df[col][row] = float(self.df[col][row])
                        except AttributeError:
                           self.df[col][row] = np.nan
                        except ValueError:
                            self.df[col][row] = np.nan
        if (col == "Region"):
            self.df[col] = self.df[col].astype(object)
        elif (col == "NSED" or col == "Coretype"):
            self.df[col] = self.df[col].astype(int)
        else:
            self.df[col] = self.df[col].astype(float)   
            
    def data_fill(self):
        #Train inputer for columns with missing vales
        #Should be run after data clean
        """Found missing values with df.isnull().sum()
        
        
        JUST PUT IN 0 INSTEAD OF THE DATA INPUTER"""
        
        missing_values = ['Signi70', 'FWHMa70', 'FWHMb070', 'FWHMb070', 'PA070', "FWHMa160",
                      'FWHMa160', 'FWHMb160', 'PA160', 'FWHMa250', 'FWHMb250', 'PA250',
                      'Signi500']

        imputer = KNNImputer(missing_values = np.nan, add_indicator = True, n_neighbors = 10)
        inputted = imputer.fit_transform(self.df[missing_values])
        i = 0
        for col in missing_values:
            self.df[col] = inputted[:,i]
            i+=1
        self.df[missing_values] = imputer.fit_transform(self.df[missing_values])
        

    def plot_hist(self):
        #Should save to file
        for col in self.df.columns:
            plt.figure()
            plt.hist(self.df[col])
            plt.ylabel(col)
        
    def scale(self, scalers):
        for scale_type in scalers:
            scaler = self.scalerTypes[scale_type]
            print(f'Scaling test and training x data using {scaler}')
            self.X = scaler.fit_transform(self.X)
            print(f'\nSummary of dataframe scaled with {scaler}:')
            self.X = pd.DataFrame(self.X)
            print(self.X.describe)
            
    def plot_heatmap(self):
        self.df.heat_map()
        
        
def main():
    #One main function to do your plotting and description
    clean = Data_Cleaner(['quantileTransform'])
    #clean.plot_hist()
    clean.plot_heatmap()
        
if __name__ == '__main__':
    main()    

    
    
    
    
    
    
    
    
    
    
    
    
    
            