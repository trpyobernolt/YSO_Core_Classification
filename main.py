#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification algorithm 
"""

from midterm_model import Model
from sklearn import preprocessing
import midterm_statistics


tests = 20
testing_average = 0
for i in range(tests):
    model1 = Model("Random Forest Classifier", [preprocessing.MaxAbsScaler()])
    #midterm_statistics.histograms()
    midterm_statistics.describe(model1)
    model1.data_engineer()
    #midterm_statistics.describe(model1)
    model1.scale_data()
    model1.split_data(0.3)
    testing_average += model1.run()
    
testing_average = testing_average / tests
print("\n\n\n Testing Average is: ", testing_average)

#X[X._____ (column) < np.percentile(_____, 99)]
#this is how you drop values over 99th percentile?



def run(size):
    model1 = Model("Random Forest Classifier", [preprocessing.MaxAbsScaler()])
    model2 = Model("Decision Tree", [preprocessing.MaxAbsScaler()])
    model3 = Model("KNN", [preprocessing.MaxAbsScaler()])
    models = []
    models.append(model1)
    models.append(model2)
    models.append(model3)
    for model in models:
        model.split_data(size)
        midterm_statistics.shapes(model)
        model.scale_data()
        score = model.run()
        print(model.label, "has an accuracy score of ", score, "\n\n\n")
        
#if __name__ == "__main__":
    run(0.3)

        
        
        
