#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification algorithm 
"""

from midterm_model import Model
from sklearn import preprocessing
import midterm_statistics


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
        
if __name__ == "__main__":
    run(0.3)

        
        
        
