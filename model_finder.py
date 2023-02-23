#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:25:11 2022

@author: troyobernolte
"""
#Related Modules
import data_reader
import data_edit
#for data handling
import pandas as pd
import numpy as np
#for stats tests
import scipy
#for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
#for saving optimized algorithms
import pickle
from joblib import dump,load
import os
#for machine learning
import scipy.sparse
import scipy.optimize
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
#for metric evaluations
from sklearn.metrics import accuracy_score, pairwise, classification_report, confusion_matrix



class MLModels:
  ##classwide dictionaries for Classifiers and for parameter dictionaries
  algorithms = {'NB':ComplementNB(), 'Perceptron':linear_model.Perceptron(), 'SVM':svm.SVC(), \
                  'SGD':linear_model.SGDClassifier(), \
                'PassiveAggressive':linear_model.PassiveAggressiveClassifier(), 'LinearDisc':discriminant_analysis.LinearDiscriminantAnalysis(), \
                'QuadDisc':discriminant_analysis.QuadraticDiscriminantAnalysis(), 'KNN':neighbors.KNeighborsClassifier(), \
                 'DeciscionTree':tree.DecisionTreeClassifier()
                }
      
  ##Parameters for each model. These are searched by GridSearchCV
  parameterDict = {"NB":{'alpha':[1e-9, 1e-6, 1e-3, 0.1, 0.5, 1, 2, 3], 'norm':[True, False]}, \
                   "Perceptron":{'penalty':[None, 'l2', 'l1'], 'alpha':[1e-9, 1e-7, 1e-5, 1e-3, 0.1], 'fit_intercept':[True, False], \
                      'shuffle':[True, False], 'n_iter_no_change':[5, 10, 15], 'class_weight':['balanced', None]}, \
                  "SVM":{'C':[0.1, 0.25, 0.5, 0.75, 1, 2, 3], 'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'rbf'], \
                      'degree':[1, 2, 3, 4], 'gamma':['scale', 'auto'], 'shrinking':[True, False], \
                      'class_weight':['balanced', None]},\
                  "LogReg":{'penalty':['l2', 'none'], 'tol':[1e-5, 1e-4, 1e-2, 1e-1], 'C':[0.25, 0.5, 1, 2], \
                      'fit_intercept':[True, False], 'class_weight':[None, 'balanced'], \
                      'solver':['newton-cg', 'lbfgs', 'sag', 'saga'], 'multi_class':['auto', 'ovr', 'multinomial']}, \
                      
                   "SGD":{'loss':['hinge', 'modified_huber'], 'penalty':['elasticnet', 'l1', 'l2'], \
                      'alpha': [1e-4, 1e-2, 0.1, 1, 2], 'shuffle':[True, False], 'learning_rate':['constant', 'optimal', 'invscaling', 'adaptive'], \
                      'class_weight':['balanced', None], 'eta0':[1e-4, 0.001]}, \
                       
                  "PassiveAggressive":{'C':[0.25, 0.5, 1, 2], 'n_iter_no_change':[5, 10, 15], 'shuffle':[True, False], \
                      'class_weight':['balanced', None]}, \
                  "LinearDisc":{'solver':['svd', 'lsqr', 'eigen'], 'shrinkage':[None, 'auto', 0.1, 0.5], 'tol':[1e-4, 1e-2, 0.1]}, \
                  "QuadDisc":{'reg_param':[0, 0,1, 0.5, 1], 'tol':[1e-4, 1e-3, 1e-2, 1e-1]}, \
                   "KNN":{'n_neighbors':[5, 10, 15, 20], 'weights':['uniform', 'distance'], \
                          'algorithm':['ball_tree', 'kd_tree', 'brute', 'auto'], 'leaf_size':[10, 20, 30, 40], 'p':[1, 2]}, \
                   "GaussProcess":{'kernel':[None, kernels.Matern(), kernels.RationalQuadratic(), kernels.ExpSineSquared(), kernels.DotProduct()], \
                                   'max_iter_predict':[100, 150, 200], 'multi_class':['one_vs_rest', 'one_vs_one']},\
                   "DeciscionTree":{'criterion':['gini', 'entropy'], 'splitter':['best','random'], \
                                    'max_depth':[None, 10, 50, 100], 'min_samples_split':[0.5, 2, 4], 'min_samples_leaf':[0.5, 1, 2], \
                                    'min_weight_fraction_leaf':[0, 0.25, 0.5], 'max_features':[None, 'auto', 'sqrt', 'log2'], \
                                    'max_leaf_nodes':[None, 10, 20], 'class_weight':[None, 'balanced']} \
                   }

  #Filepath for .joblib files that contain optimized classifiers
  #CHANGE THIS
  savePath = "Data/Optimized Classifiers/"


  ##constructor method assigns X and Y data to be utilized by algorithms
  def __init__(self, X, Y):
    self.x = X
    self.y = Y

  ##method splits the data into a training set and testing set based on parameter
  def splitTestTrain(self, ratio):
    #split data and print shape of train and test sets
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = ratio)
    print(f'Shape of original Dataframe: {self.x.shape} {self.y.shape} \n\
    Shape of training data: {self.x_train.shape} {self.y_train.shape} \n\
    Shape of testing data: {self.x_test.shape} {self.y_test.shape}')


  #Method utilizes Scikitlearns GridSearchCV object to construct best parameters settings for any given classification algorithm
  def findBestParams(self):

    for algo in (self.algorithms.keys()):
        if not(os.path.isfile(f'Data/Optimized Classifiers/{algo}.joblib')):
            
            ##instantiating classifier and GridSearch with classifier
            myClassifier = self.algorithms[algo]
            myGrid = GridSearchCV(myClassifier, self.parameterDict[algo], scoring = 'accuracy', cv = 10)
            myGrid.fit(self.x_train, self.y_train)
            optimizedClassifier = myGrid.best_estimator_
            print(f'Optimized Parameters for {algo}: \n {optimizedClassifier.get_params}')
            print(f'Classifying Data with optimized {algo}')

             ##utilize the best estimator on the holdout data
            self.classifyData(algo, optimizedClassifier)
             ##saving the best estimator into a joblibFile
            self.saveAlgo(algo, optimizedClassifier)

  ##method classifies data and prints results of algorithms. 
  ##Algorithm parameters accepts SciKitLearn Estimator Objects (assumed to be parametricized)
  def classifyData(self, algoName, algorithm):
    
    myCalibrator = CalibratedClassifierCV(algorithm)
    myCalibrator.fit(self.x_train, self.y_train)
    self.y_pred = myCalibrator.predict(self.x_test)
    self.showResults(algoName)

    ##method analyzes predicted values generated from classifyData. Prints Confusion matrix, classification report, and overall accuracy of the algorithm
  def showResults(self, algoName):
      print(f'Confusion Matrix and full Classification Report of {algoName}: \n{confusion_matrix(self.y_test, self.y_pred)}')
      print(classification_report(self.y_test, self.y_pred)) 

      # Evaluate label (subsets) accuracy
      print(f'Overall Accuracy of {algoName}: {accuracy_score(self.y_test, self.y_pred)}\n')
      
  def getResults(self, algoName):
      return classification_report(self.y_test, self.y_pred)

  #Method to utilize saved algorithms instead of rerunning GridSearch
  def utilizeBest(self):
    """Change to reutrn a dictionary with the model and its accuracy.
    """
    best_list = []
    class_reports = []
    for algo in self.algorithms.keys():
        myAlgo = self.loadAlgo(algo)
        self.classifyData(algo, myAlgo)
        best_list.append(myAlgo)
        class_reports.append(self.getResults(myAlgo))
    return best_list, class_reports

  #saves algorithms as .joblib files
  def saveAlgo(self, algoName, algorithm):
    saveFile = open(self.savePath + algoName + '.joblib', 'wb')
    dump(algorithm, saveFile)
    saveFile.close()

  #loads and returns an algorithm with its best parameters (currently unused)
  def loadAlgo(self, algoName):
    saveFile = open(self.savePath + algoName + '.joblib', 'rb')
    loadedAlgo = load(saveFile)
    return loadedAlgo

  def returnBest(self):
    best = []
    for algo in self.algorithms.keys():
        best.append(self.algo)
    return best






def train():
    cleaner = data_edit.Data_Cleaner(['quantileTransform'])
    myMachine = MLModels(cleaner.X, cleaner.Y) 
    myMachine.splitTestTrain(0.3)
    myMachine.findBestParams()
    
def evaluate():
    cleaner = data_edit.Data_Cleaner(['quantileTransform'])
    myMachine = MLModels(cleaner.X, cleaner.Y) 
    myMachine.splitTestTrain(0.3)
    myMachine.utilizeBest()
    
