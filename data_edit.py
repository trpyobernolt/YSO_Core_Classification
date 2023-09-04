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
from scipy.stats import pointbiserialr


from sklearn.decomposition import PCA
import os



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
        
    def __init__(self, scalers, PCA):
        self.PCA = PCA
        self.df = data_reader.main()
        self.data_clean()
        missing_values_by_region_to_txt(self.df)
        graph_nans(self.df, 'FWHMa160')
        #self.data_fill_zeros()
        self.data_fill()
        self.X = self.df.drop(['Coretype','Region', 'Signi70', 'Signi160',
                               'Signi250', 'Signi350', 'Signi500', 'NSED'], axis=1)
        self.Y = self.df['Coretype']
        data_with_region = self.df.drop(['Signi70', 'Signi160',
                               'Signi250', 'Signi350', 'Signi500', 'NSED'], axis=1)
        data_with_region.to_csv('Data/data_with_region.csv', index=False)
        self.df = self.df.drop(['Region', 'Signi70', 'Signi160',
                               'Signi250', 'Signi350', 'Signi500', 'NSED'], axis=1)
        self.scale(scalers)
        self.df.to_csv('Data/clean_data.csv', index=False)
        
    def data_clean(self):
        """Cleans data. Should be run before anything else"""
        for col in self.df.select_dtypes(include='object').columns:
            for row in range(self.df.shape[0]):
                if not(col == "Region" or col == "Coretype"):
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
        elif (col == "Coretype"):
            self.df[col] = self.df[col].astype(int)
        else:
                self.df[col] = self.df[col].astype(float)
            
    def data_fill(self):
        #Train inputer for columns with missing vales
        #Should be run after data clean
        """Found missing values with df.isnull().sum()"""
        
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
        
    def data_fill_zeros(self):
        """Fills missing values with zeros. Should be run after data_clean."""
        self.df.fillna(0, inplace=True)
        
    

        
        
    def scale(self, scalers, n_components=3):
        if self.PCA == True:   
            pca = PCA(n_components=n_components)
            self.X = pca.fit_transform(self.X)
        for scale_type in scalers:
            scaler = self.scalerTypes[scale_type]
            print(f'Scaling test and training x data using {scaler}')
            self.X = scaler.fit_transform(self.X)
            print(f'\nSummary of dataframe scaled with {scaler}:')
        self.X = pd.DataFrame(self.X)
        print(self.X.describe)
            
        
        
def graph_nans(df, feature):
    # Calculate the number of missing values in the Signi70 column by region
    missing_values_by_region = df[df[feature].isnull()].groupby('Region').size()

    # Bar plot settings
    barWidth = 0.85
    regions = missing_values_by_region.index
    bars = missing_values_by_region.values
    colors = ['#FFA07A', '#87CEFA', '#FFDAB9', '#9370DB', '#98FB98', '#FFC0CB', '#B0C4DE']

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(regions, bars, color=colors, width=barWidth, edgecolor='white')
    
    # Set the title and labels
    ax.set_title(f'Missing Values in {feature} Column by Region', fontsize=16, fontweight='bold')
    ax.set_xlabel('Region')
    ax.set_ylabel('Number of Missing Values')

    # Create the directory if it doesn't exist
    save_directory = 'Graphs/Bars'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save the figure as a high-quality PDF
    plt.tight_layout()
    plt.savefig(f'{save_directory}/Missing_Values_by_Region.pdf', format='pdf', dpi=300)
    plt.show()
    plt.close(fig)

def missing_values_by_region_to_txt(df):
    # Group the DataFrame by region and count the number of missing values for each column
    missing_values_by_region = df.isnull().groupby(df['Region']).sum()

    # Create the directory if it doesn't exist
    save_directory = 'Txt_files'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the results to a .txt file
    with open(f'{save_directory}/Missing_Values_by_Region.txt', 'w') as file:
        file.write("Missing Values by Region\n")
        file.write("="*30 + "\n")
        for region in missing_values_by_region.index:
            file.write(f"{region}\n")
            file.write("-" * 30 + "\n")
            region_data = missing_values_by_region.loc[region]
            region_has_missing_values = False
            for column, missing_count in region_data.items():
                if missing_count > 0:
                    region_has_missing_values = True
                    file.write(f"{column}: {missing_count}\n")
                    if region == "Cepheus" and column == "Signi70":
                        cepheus_df = df[df['Region'] == "Cepheus"]
                        missing_values_by_coretype = cepheus_df[cepheus_df['Signi70'].isnull()].groupby('Coretype').size()
                        for coretype, count in missing_values_by_coretype.items():
                            file.write(f"  Coretype {coretype}: {count}\n")
            if not region_has_missing_values:
                file.write("No missing values\n")
            file.write("="*30 + "\n" + "\n")
            
def point_biserial_correlations(df, categorical_var):
    # Calculate the Point-Biserial Correlation for each category
    unique_categories = df[categorical_var].unique()
    continuous_vars = df.columns.drop(categorical_var)

    # Create the directory if it doesn't exist
    os.makedirs("Txt_files", exist_ok=True)

    # Open the file to write the results
    with open("Txt_files/Correlations/Point_Biserial_Correlations.txt", "w") as file:
        for category in unique_categories:
            file.write(f"Category: {category}\n")
            for continuous_var in continuous_vars:
                if continuous_var != "Region":
                    binary_cat = (df[categorical_var] == category).astype(int)
                    correlation, _ = pointbiserialr(df[continuous_var], binary_cat)
                    file.write(f"{continuous_var}: {correlation:.4f}\n")
            file.write("\n")
            
def point_biserial_correlations_sorted(df, categorical_var):
    # Calculate the Point-Biserial Correlation for each category
    unique_categories = df[categorical_var].unique()
    continuous_vars = df.columns.drop(categorical_var)

    # Create the directory if it doesn't exist
    os.makedirs("Txt_files", exist_ok=True)

    # Open the file to write the sorted results
    with open("Txt_files/Correlations/Point_Biserial_Correlations_Sorted.txt", "w") as file:
        for category in unique_categories:
            file.write(f"Category: {category}\n")
            
            # Prepare the list for storing correlation pairs
            corr_pairs = []

            for continuous_var in continuous_vars:
                if continuous_var != "Region":
                    binary_cat = (df[categorical_var] == category).astype(int)
                    correlation, _ = pointbiserialr(df[continuous_var], binary_cat)
                    corr_pairs.append((continuous_var, correlation))

            # Sort the correlation pairs by the absolute value of their correlation
            sorted_corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[1]), reverse=True)

            for continuous_var, correlation in sorted_corr_pairs:
                file.write(f"{continuous_var}: {correlation:.4f}\n")

            file.write("\n")


def bivariate_correlations(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr(method='pearson')

    # Create the directory if it doesn't exist
    os.makedirs("Txt_files", exist_ok=True)

    # Open the file to write the results
    with open("Txt_files/Correlations/Bivariate Correlations.txt", "w") as file:
        # Iterate through the correlation matrix, skipping duplicate pairs
        for i, row_name in enumerate(corr_matrix.index[:-1]):
            for j, col_name in enumerate(corr_matrix.columns[i+1:]):
                correlation = corr_matrix.at[row_name, col_name]
                file.write(f"{row_name} - {col_name}: {correlation:.4f}\n")
                
def bivariate_correlations_sorted(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr(method='pearson')

    # Prepare the list for storing correlation pairs
    corr_pairs = []

    # Iterate through the correlation matrix, skipping duplicate pairs
    for i, row_name in enumerate(corr_matrix.index[:-1]):
        for j, col_name in enumerate(corr_matrix.columns[i+1:]):
            correlation = corr_matrix.at[row_name, col_name]
            corr_pairs.append((row_name, col_name, correlation))

    # Sort the correlation pairs by the absolute value of their correlation
    sorted_corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

    # Create the directory if it doesn't exist
    os.makedirs("Txt_files", exist_ok=True)

    # Open the file to write the sorted results
    with open("Txt_files/Correlations/Bivariate_Correlations_Sorted.txt", "w") as file:
        for row_name, col_name, correlation in sorted_corr_pairs:
            file.write(f"{row_name} - {col_name}: {correlation:.4f}\n")




def main():
    #One main function to do your plotting and description
    clean = Data_Cleaner(['quantileTransform'], PCA=False)
    #df = pd.read_csv('Data/clean_data.csv')
    #point_biserial_correlations(df, 'Coretype')
    #point_biserial_correlations_sorted(df, 'Coretype')
    #bivariate_correlations(df)
    #bivariate_correlations_sorted(df)
    #clean.plot_hist()
    #clean.plot_heatmap()
        
if __name__ == '__main__':
    main()    

    
    
    
    
    
    
    
    
    
    
    
    
    
            