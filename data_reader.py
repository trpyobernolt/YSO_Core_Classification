#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:53:43 2022

@author: troyobernolte

This code is meant to take data from all given sources, give them binary
classifications, and save the file as a csv
"""

import pandas as pd

#Cepheus Table
dataset = pd.read_csv('Data/cepheus_tablea1.tsv',delimiter=';',comment='#') 
X1 = dataset[['Signi70', 'Sp70',
       'e_Sp70', 'Sp70/Sbg70', 'Sconv70', 'Stot70', 'e_Stot70', 'FWHMa70',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/NbgH2', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED', 'CSARflag']]

    
Y1 = dataset['Coretype']

#ophiuchus_tablea1
dataset = pd.read_csv('Data/ophiuchus_tablea1.tsv', delimiter=';',comment='#')
X2 = dataset[['Signi070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED']]

Y2 = dataset['Coretype']

#taurus_table1
dataset = pd.read_csv('Data/taurus_table1.tsv', delimiter=';',comment='#')
X3 = dataset[['sig70', 'I70peak', 'e_I70peak',
       'C70', 'I70conv', 'S70', 'e_S70', 'a70', 'b70', 'PA70', 'sig160',
      'I160peak', 'e_I160peak', 'C160', 'I160conv', 'S160', 'e_S160', 'a160',
       'b160', 'PA160', 'sig250', 'I250peak', 'e_I250peak', 'C250', 'I250conv',
      'S250', 'e_S250', 'a250', 'b250', 'PA250', 'sig350', 'I350peak',
     'e_I350peak', 'C350', 'I350conv', 'S350', 'e_S350', 'a350', 'b350',
     'PA350', 'sig500', 'I500peak', 'e_I500peak', 'C500', 'S500', 'e_S500',
       'a500', 'b500', 'PA500', 'sigNH2', 'NH2peak', 'CNH2', 'NH2conv',
       'NH2bg', 'aNH2', 'bNH2', 'PANH2', 'Nsed']]
Y3 = dataset['CType']



#Corona Australia
dataset = pd.read_csv('Data/corona_australia_table1.tsv', delimiter=';',comment='#')
X4 = dataset[['Signi070', 'Sp070', 'e_Sp070',
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
        
Y4 = dataset['Coretype']

#lupus_tablea1
dataset = pd.read_csv('Data/lupus_tablea1.tsv', delimiter=';',comment='#')
X5 = dataset[['runNo', 'CoreName', 'RAJ2000', 'DEJ2000', 'Sig070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Sig160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Sig250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Sig350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Sig500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigNH2', 'NH2p', 'NH2p/NH2bg', 'NH2conv', 'NH2bg', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED', 'CuTExflag']]
Y5 = dataset['Coretype']

#aquilia_table1
dataset = pd.read_csv('Data/aquila_table1.tsv', delimiter=';',comment='#')
X6 = dataset[['Signi070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED', 'CSARflag']]
Y6 = dataset['Coretype']

#orionb_table1
dataset = pd.read_csv('Data/orionb_table1.tsv', delimiter=';',comment='#')
X7 = dataset[['Signi070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED']]
    
Y7 = dataset['Coretype']

#Concat all data to merge to one dataframe. Ignore index to have a discrete
# [0, n-1] indexing
X = pd.concat([X1, X2, X3, X4, X5, X6, X7], ignore_index=True)
Y = pd.concat([Y1, Y2, Y3, Y4, Y5, Y6, Y7], ignore_index=True)
Y_raw = Y.copy()

"""Original set of labels was:
    Returns the labels of:
        {'', 'starless', 1, 3, 2, 4, 'protostellar', '-1', '1', '3', '2', 
         'prestellar', 0, '--', '4'}
"""

#Engineer data to binary classification
#Setting 0 for prestellar cores and 1 for protostellar cores

#I am sure there is a better way to do this, but setting a list of rows to drop
drop_list = []

#Iterate over rows
for row in range(len(Y)):
    if isinstance(Y.iloc[row], str):
        Y.iloc[row] = str.strip(Y.iloc[row])
        #drop the labels that we have missing data for
        if ( Y.iloc[row] == '' or Y.iloc[row] == '--'):
            drop_list.append(row)
        #Set a positive label for protostellar classification
        elif Y.iloc[row] == 'protostellar':
            Y.iloc[row] = 1
        else:
            Y.iloc[row] = 0
    elif isinstance(Y.iloc[row], int):
        if (Y.iloc[row] == -1):
            drop_list.append(row)
        if Y.iloc[row] > 2:
            Y.iloc[row] = 1
        else:
            Y.iloc[row] = 0
            
for row in drop_list:
    X = X.drop(row)
    Y = Y.drop(row)

#Reset the indexes to be able to easily with with data later on
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

X.to_csv('Data/X_vals.csv', index=False)
Y.to_csv('Data/Y_vals.csv', index=False)
























