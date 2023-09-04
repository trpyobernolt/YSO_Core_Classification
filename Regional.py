#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:47:29 2023

@author: troyobernolte
"""

import scipy.linalg
from scipy.sparse.linalg import cg, lsqr
from scipy.optimize import minimize
from sklearn import preprocessing, model_selection, feature_selection, \
ensemble, linear_model, metrics, decomposition, svm, naive_bayes, discriminant_analysis, \
kernel_ridge, neighbors, gaussian_process, tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, \
QuantileTransformer
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import make_column_selector as selector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import class_weight
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import MinMaxScaler

#for metric evaluations
from sklearn.metrics import accuracy_score, pairwise, classification_report, confusion_matrix
from joblib import dump,load


import pandas as pd
from model_finder import MLModels
from collections import defaultdict
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import json


algorithms = {'NB':ComplementNB(), 'Perceptron':linear_model.Perceptron(), 'SVM':svm.SVC(), \
                  'SGD':linear_model.SGDClassifier(), \
                'PassiveAggressive':linear_model.PassiveAggressiveClassifier(), 
                'LinearDisc':discriminant_analysis.LinearDiscriminantAnalysis(), \
                'QuadDisc':discriminant_analysis.QuadraticDiscriminantAnalysis(),
                'KNN':neighbors.KNeighborsClassifier(), \
                 'DeciscionTree':tree.DecisionTreeClassifier()
                }
    

    
def loadAlgo(algoName, filename):
    saveFile = open("Data/" + filename + "/" + algoName + '.joblib', 'rb')
    loadedAlgo = load(saveFile)
    return loadedAlgo


def evaluate_trained_models(x_test, y_test, algorithms, filename):
    class_reports = {}
    
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1': make_scorer(f1_score, average='weighted')}
    scoring = {'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1': make_scorer(f1_score, average='weighted'),
               'accuracy': make_scorer(accuracy_score)}
    
    for algo_name in algorithms.keys():
        model = loadAlgo(algo_name, filename)
        cv_results = cross_validate(model, x_test, y_test, cv=10, scoring=scoring, return_train_score=False)
        class_reports[algo_name] = {
            'mean_accuracy': cv_results['test_accuracy'].mean(),
            'mean_precision': cv_results['test_precision'].mean(),
            'mean_recall': cv_results['test_recall'].mean(),
            'mean_f1': cv_results['test_f1'].mean()
        }
    return class_reports

def format_float(value):
    return round(value, 2)

def write_results_to_file(results, filename):
    with open(filename, "w") as file:
        for region, region_result in results.items():
            file.write(f"results_list_{region}_Unbalanced = []\n")
            file.write('"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""\n')
            for algo, class_report in region_result.items():
                precision = round(class_report["mean_precision"], 2)
                recall = round(class_report["mean_recall"], 2)
                f1 = round(class_report["mean_f1"], 2)
                accuracy = round(class_report["mean_accuracy"], 2)
                file.write(f'results_list_{region}_Unbalanced.append(["{algo}", {precision}, {recall}, {f1}, {accuracy}])\n')
            file.write("res_df_{}_Unbalanced = pd.DataFrame(results_list_{}_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])\n\n".format(region, region))



            
results = defaultdict(dict)

data = pd.read_csv("Data/clean_data.csv")


def regional_results(balanced_lib_file):

    training_set = pd.read_csv(f"Data/{balanced_lib_file}/xtrain")

    not_trained = data.drop(training_set["Unnamed: 0"], axis=0)
    
    regions = data.Region.unique()
    
    
    for region in regions:
        df = not_trained.loc[not_trained['Region'] == region]
        y = df['Coretype'].astype('category')
        columns_to_keep = [col for col in df.columns if col not in 
                           ['Coretype','Region', 'Signi70', 'Signi160',
                            'Signi250', 'Signi350', 'Signi500', 'NSED']]
        x = df.loc[:, columns_to_keep]
        x = x.applymap(lambda x: 0 if x < 0 else x)
        
        region_results = evaluate_trained_models(x, y, algorithms, balanced_lib_file)
        results[region] = region_results
    
    # Save results to a file
    output_filename = f"Txt_files/{balanced_lib_file}_results_by_region.txt"
    write_results_to_file(results, output_filename)
        
