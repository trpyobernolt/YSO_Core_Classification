#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification algorithm 
"""

from midterm_model import Model
from sklearn import preprocessing
import midterm_statistics

        
def testing_all_methods(sample_size, sc):
    classifiers = ["KNN", "Support Vector Machine", "Gaussian Process Classifier",
              "Random Forest Classifier", "MLP Classifier", "Ada Boost Classifier",
              "GnB", "Perceptron", "Decision Tree"]
    best_accuracy = -1
    best_model = None
    scalar_type = sc
    #Run each classifier
    for classifier in classifiers:
        total_accuracy = 0
        #Test 100 times and take an average
        for iteration in range(100):
            model = Model(classifier, scalar_type)
            model.split_data(sample_size)
            model.scale_data()
            total_accuracy += model.run()
        avg_accuracy = total_accuracy/100
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_model = model
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("Best model is ", best_model.label, "With an average accuracy of ",
          best_accuracy)
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("--------------------------------------------")
    
def testing_one_method(method, sample_size, sc, tests):
    """Takes in one method, a sample size, and a scalar and returns the average
    performance of accuracy over a given number of runs"""
    scalar_type = sc
    total_accuracy = 0
    for iteration in range(tests):
        model = Model(method, scalar_type)
        model.split_data(sample_size)
        model.scale_data()
        total_accuracy += model.run()
    avg_accuracy = total_accuracy / tests
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("--------------------------------------------")
    print(method, "Has an average accuracy of ", avg_accuracy)
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("--------------------------------------------")
    

#testing(0.3, [])
#testing(0.3, [preprocessing.MaxAbsScaler()])
#testing_one_method("Random Forest Classifier", 0.3, [preprocessing.RobustScaler()], 100)

def optimization_RF(one, two, three):
    model = Model("Random Forest Classifier", [preprocessing.RobustScaler()], one, two, three)
    model.split_data(0.3)
    model.scale_data()
    return model.run()

best_score = -1
best_max_depth = None
best_n_estimators = None
best_max_features = None
for i in range(10):
    for k in range(20, 50):
        for z in range(10):
            avg_score = 0
            for iterations in range(10):
                avg_score += optimization_RF(i+1, k+1, z+1)
            if avg_score > best_score:
                best_max_depth = i
                best_n_estimators = k
                best_max_features = z
                best_score = avg_score
                
print("Best Max Depth is: ", best_max_depth)
print("Best n estimators is: ", best_n_estimators)
print("Best max features is: ", best_max_features)
print("This gives an accuracy of: ", best_score())


"""
Max Abs results in accuracy of 7.4639

"""

"""
model = Model("Random Forest Classifier", [preprocessing.MaxAbsScaler()])
midterm_statistics.describe(model)
model.split_data(0.3)
model.scale_data()
model.run()
"""
        
""" Code for optimization of Random Forest. Results in:
    Best Max Depth is:  3
    Best n estimators is:  21
    Best max features is:  9
    
    
def optimization_RF(one, two, three):
    model = Model("Random Forest Classifier", 'robust', one, two, three)
    model.split_data(0.3)
    model.scale_data()
    return model.run()

best_score = -1
best_max_depth = None
best_n_estimators = None
best_max_features = None
for i in range(10):
    for k in range(50):
        for z in range(10):
            score = optimization_RF(i+1, k+1, z+1)
            if score > best_score:
                best_max_depth = i
                best_n_estimators = k
                best_max_features = z
                best_score = score
                
print("Best Max Depth is: ", best_max_depth)
print("Best n estimators is: ", best_n_estimators)
print("Best max features is: ", best_max_features)
print("This gives an accuracy of: ", best_score())
"""

        
        
