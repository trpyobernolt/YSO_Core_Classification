#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification algorithm 

"""

from model_finder import MLModels
from balanced_model_finder import Balanced_MLModels
import data_edit
import balanced_data_edit
from kfold_model_finder import Kfold_MLModels
from Regional import regional_results


def main():
    #Run this code for unbalanced optimization
    """cleaner = data_edit.Data_Cleaner(['quantileTransform', 'minmax'], PCA=False)
    myMachine = MLModels(cleaner.X, cleaner.Y) 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.utilizeBest()
    return list_best"""

    #Run this code to get metrics for UNBALANCED k-fold validation
    cleaner = data_edit.Data_Cleaner(['quantileTransform', 'minmax'], PCA=False)
    myMachine = Kfold_MLModels(cleaner.X, cleaner.Y, "Unbalanced Optimized Classifiers") 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.utilizeBest(cv_folds=10)
    #return list_best

    #Run this code to find unbalanced optimization
    """cleaner = data_edit.Data_Cleaner(['quantileTransform', 'minmax'], PCA=False)
    myMachine = Kfold_MLModels(cleaner.X, cleaner.Y, "Unbalanced Optimized Classifiers") 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.findBestParams()
    return list_best"""


    #Run this code to find balanced optimization
    """cleaner = balanced_data_edit.Data_Cleaner(['quantileTransform', 'minmax'], 200)
    myMachine = Balanced_MLModels(cleaner.X, cleaner.Y)
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.findBestParams()
    return list_best"""
    
    #Run this code to get metrics for BALANCED k-fold validation
    cleaner = balanced_data_edit.Data_Cleaner(['quantileTransform', 'minmax'], 200)
    myMachine = Kfold_MLModels(cleaner.X, cleaner.Y, "Balanced Optimized Classifiers") 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.utilizeBest(cv_folds=10)
    #return list_best

    #Run this code for balanced regional evaluation. Will be written to .txt file
    regional_results("Balanced Optimized Classifiers")
    
    #Run this code for unbalanced regional evaluation. Will be written to .txt file
    regional_results("Optimized Classifiers")

    
    
    #Runs balanced classifiers on full dataset.
    """cleaner = data_edit.Data_Cleaner(['quantileTransform', 'minmax'], 200)
    myMachine = Kfold_MLModels(cleaner.X, cleaner.Y, "Balanced Optimized Classifiers") 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.utilizeBest()
    return list_best"""

    #Run this code to find best unbalanced model with PCA
    """cleaner = data_edit.Data_Cleaner(['quantileTransform', 'minmax'], PCA=True)
    myMachine = Kfold_MLModels(cleaner.X, cleaner.Y, "Unbalanced Optimized Classifiers with PCA") 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.findBestParams()
    return list_best"""

    #Run this code to UTILIZE best unbalanced model with PCA
    """cleaner = data_edit.Data_Cleaner(['quantileTransform', 'minmax'], PCA=True)
    myMachine = Kfold_MLModels(cleaner.X, cleaner.Y, "Unbalanced Optimized Classifiers with PCA") 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.utilizeBest(cv_folds=10)
    return list_best"""

def GraphBest():
    best_list = main()
    

if __name__ == '__main__':
    main()
        
        
        
