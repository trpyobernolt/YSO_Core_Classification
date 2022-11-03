
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:53:43 2022
@author: troyobernolte
This code is meant to take data from all given sources, give them binary
classifications, and save the file as a csv
"""

import pandas as pd



class RawSEDData:

    """"Read raw table and do some pre-processing"""

    def __init__(self):
        
        #Get the raw data columns from each individual assignment
       raw1 = set(['Signi70', 'Sp70',
              'e_Sp70', 'Sp70/Sbg70', 'Sconv70', 'Stot70', 'e_Stot70', 'FWHMa70',
              'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
              'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
              'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
              'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
              'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
              'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
              'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
              'SigniNH2', 'NpH2', 'NpH2/NbgH2', 'NconvH2', 'NbgH2', 'FWHMaNH2',
              'FWHMbNH2', 'PANH2', 'NSED', 'CSARflag'])
       
       raw2 = set(['Signi070', 'Sp070', 'e_Sp070',
              'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
              'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
              'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
              'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
              'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
              'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
              'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
              'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
              'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
              'FWHMbNH2', 'PANH2', 'NSED'])
       
       raw3 = set(['sig70', 'I70peak', 'e_I70peak',
              'C70', 'I70conv', 'S70', 'e_S70', 'a70', 'b70', 'PA70', 'sig160',
             'I160peak', 'e_I160peak', 'C160', 'I160conv', 'S160', 'e_S160', 'a160',
              'b160', 'PA160', 'sig250', 'I250peak', 'e_I250peak', 'C250', 'I250conv',
             'S250', 'e_S250', 'a250', 'b250', 'PA250', 'sig350', 'I350peak',
            'e_I350peak', 'C350', 'I350conv', 'S350', 'e_S350', 'a350', 'b350',
            'PA350', 'sig500', 'I500peak', 'e_I500peak', 'C500', 'S500', 'e_S500',
              'a500', 'b500', 'PA500', 'sigNH2', 'NH2peak', 'CNH2', 'NH2conv',
              'NH2bg', 'aNH2', 'bNH2', 'PANH2', 'Nsed'])
       
       raw4 = set(['Signi070', 'Sp070', 'e_Sp070',
                      'Sp/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070', 'FWHMb070',
                      'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp/Sbg160', 'Sconv160',
                      'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160', 'Signi250',
                      'Sp250', 'e_Sp250', 'Sp/Sbg250', 'Sconv250', 'Stot250', 'e_Stot250',
                      'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350', 'e_Sp350',
                      'Sp/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350', 'FWHMa350', 'FWHMb350',
                      'PA350', 'Signi500', 'Sp500', 'e_Sp500', 'Sp/Sbg500', 'Stot500',
                      'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500', 'SigniNH2', 'NpH2',
                      'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2', 'FWHMbNH2', 'PANH2', 'NSED',
                      'CSARflag', 'CUTEXflag'])
    
       raw5 = set(['runNo', 'CoreName', 'RAJ2000', 'DEJ2000', 'Sig070', 'Sp070', 'e_Sp070',
              'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
              'FWHMb070', 'PA070', 'Sig160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
              'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
              'Sig250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
              'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Sig350', 'Sp350',
              'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
              'FWHMa350', 'FWHMb350', 'PA350', 'Sig500', 'Sp500', 'e_Sp500',
              'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
              'SigNH2', 'NH2p', 'NH2p/NH2bg', 'NH2conv', 'NH2bg', 'FWHMaNH2',
              'FWHMbNH2', 'PANH2', 'NSED', 'CuTExflag'])
       
       raw6 = set(['Signi070', 'Sp070', 'e_Sp070',
              'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
              'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
              'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
              'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
              'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
              'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
              'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
              'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
              'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
              'FWHMbNH2', 'PANH2', 'NSED', 'CSARflag'])
       
       
       raw7 = set(['Signi070', 'Sp070', 'e_Sp070',
              'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
              'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
              'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
              'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
              'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
              'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
              'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
              'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
              'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
              'FWHMbNH2', 'PANH2', 'NSED'])
       
       #Take the intersection of the sets to only get data in common
       common_elements = raw1.intersection(raw2, raw3, raw4, raw5, raw6, raw7)
       
       #Cepheus Table
       dataset = pd.read_csv('Data/cepheus_tablea1.tsv',delimiter=';',comment='#') 
       self.X1 = dataset[common_elements]

       self.Y1 = dataset['Coretype']

       #ophiuchus_tablea1
       dataset = pd.read_csv('Data/ophiuchus_tablea1.tsv', delimiter=';',comment='#')
       self.X2 = dataset[common_elements]

       self.Y2 = dataset['Coretype']

       #taurus_table1
       dataset = pd.read_csv('Data/taurus_table1.tsv', delimiter=';',comment='#')
       self.X3 = dataset[common_elements]
       self.Y3 = dataset['CType']

       #Corona Australia
       dataset = pd.read_csv('Data/corona_australia_table1.tsv', delimiter=';',comment='#')
       self.X4 = dataset[common_elements]
       self.Y4 = dataset['Coretype']
       #data-specific row engineering. Cannot be done post-merge
                   
       #lupus_tablea1
       dataset = pd.read_csv('Data/lupus_tablea11_14.tsv', delimiter=';',comment='#')
       self.X5 = dataset[common_elements]
       self.Y5 = dataset['Coretype']

       #aquilia_table1
       dataset = pd.read_csv('Data/aquila_table1.tsv', delimiter=';',comment='#')
       self.X6 = dataset[common_elements]
       self.Y6 = dataset['Coretype']

       #orionb_table1
       dataset = pd.read_csv('Data/orionb_table1.tsv', delimiter=';',comment='#')
       self.X7 = dataset[common_elements]

       self.Y7 = dataset['Coretype']
       
       print("X Columns that are used are: ", common_elements)

    def cleanData(self):    
       """Perform cleaning here"""

       #I am sure there is a better way to do this, but setting a list of rows to drop
       drop_list = []
       
       #Cepheus Table Data Cleaning
       for row in range(len(self.Y1)):
           self.Y1.iloc[row] = str.strip(self.Y1.iloc[row])
           if (self.Y1.iloc[row] == "protostellar"):
               self.Y1.iloc[row] = 3
           elif (self.Y1.iloc[row] == "prestellar"):
               self.Y1.iloc[row] = 2
           elif (self.Y1.iloc[row] == "starless"):
               self.Y1.iloc[row] = 1
       print("Cepheus Y labels: ", set(self.Y1))
       drop_list = []
       
       #Ophiuchus Data Clearning
       for row in range(len(self.Y2)):
           self.Y2.iloc[row] = str.strip(self.Y2.iloc[row])
           if (self.Y2.iloc[row] == "protostellar"):
               self.Y2.iloc[row] = 3
           elif (self.Y2.iloc[row] == "prestellar"):
               self.Y2.iloc[row] = 2
           elif (self.Y2.iloc[row] == "starless"):
               self.Y2.iloc[row] = 1
       print("Ophiuchus Y labels: ", set(self.Y2))
       drop_list = []
       
       #Taurus Data Cleaning
       for row in range(len(self.Y3)):
           if (self.Y3.iloc[row] == 3):
               drop_list.append(row)
           elif (self.Y3.iloc[row] == 4):
               self.Y3.iloc[row] = 3
       for row in drop_list:
           self.X3 = self.X3.drop(row)
           self.Y3 = self.Y3.drop(row)
       print("Taurus Y labels: ", set(self.Y3))
       drop_list = []
       
       #Corona Australia Data Cleaning
       drop_list.append(0)
       drop_list.append(1)
       for row in range(len(self.Y4)):
           self.Y4.iloc[row] = str.strip(self.Y4.iloc[row])
           if (self.Y4.iloc[row] == "-1" or self.Y4.iloc[row] == "3"):
                drop_list.append(row)
           if self.Y4.iloc[row] == "4":
                self.Y4.iloc[row] = 3
           if (self.Y4.iloc[row] ==  "1" or self.Y4.iloc[row] == "2"):
               self.Y4.iloc[row] = int(self.Y4.iloc[row])
        #drop rows with unwanted data
       for row in drop_list:
            self.X4 = self.X4.drop(row)
            self.Y4 = self.Y4.drop(row)
       print("Corona Y labels: ", set(self.Y4))
       drop_list = []
       
       #Lupus Data Cleaning
       for row in range(len(self.Y5)):
           self.Y5.iloc[row] = str.strip(self.Y5.iloc[row])
           if (self.Y5.iloc[row] == "protostellar"):
               self.Y5.iloc[row] = 3
           elif (self.Y5.iloc[row] == "prestellar"):
               self.Y5.iloc[row] = 2
           elif (self.Y5.iloc[row] == "starless"):
               self.Y5.iloc[row] = 1
       print("Lupus Y labels: ", set(self.Y2))
       drop_list = []
       
       #Aquilia Data Cleaning
       for row in range(len(self.Y6)):
           self.Y6.iloc[row] = str.strip(self.Y6.iloc[row])
           if (self.Y6.iloc[row] == "protostellar"):
               self.Y6.iloc[row] = 3
           elif (self.Y6.iloc[row] == "prestellar"):
               self.Y6.iloc[row] = 2
           elif (self.Y6.iloc[row] == "starless"):
               self.Y6.iloc[row] = 1
       print("Aquilia Y labels: ", set(self.Y2))
       drop_list = []
       
       #Orion-Specific Data Cleaning             
       for row in range(len(self.Y7)):
           if isinstance(self.Y7.iloc[row], str):
               self.Y7.iloc[row] = str.strip(self.Y7.iloc[row])
               #drop the labels that we have missing data for
               if ( self.Y7.iloc[row] == '' or self.Y7.iloc[row] == '--'):
                   drop_list.append(row)
           if (self.Y7.iloc[row] == 0):
               drop_list.append(row)
                    
       #drop rows with no data
       for row in drop_list:
            self.X7 = self.X7.drop(row)
            self.Y7 = self.Y7.drop(row)
       print("Orion Y labels: ", set(self.Y7))

       X = pd.concat([self.X1, self.X2, self.X3, self.X4, self.X5, 
                      self.X6, self.X7], ignore_index=True)
       Y = pd.concat([self.Y1, self.Y2, self.Y3, self.Y4, self.Y5, 
                      self.Y6, self.Y7], ignore_index=True)
       #Reset the indexes to be able to easily deal with data later on
       X = X.reset_index(drop=True)
       Y = Y.reset_index(drop=True)

       X.to_csv('Data/X_vals.csv', index=False)
       Y.to_csv('Data/Y_vals.csv', index=False)
       return X, Y

def main():
   """Main class"""
   X_data, y_data = RawSEDData().cleanData()
   return X_data, y_data

if __name__ == '__main__':
    main()



















