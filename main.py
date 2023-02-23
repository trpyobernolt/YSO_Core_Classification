#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification algorithm 

"""

from model_finder import MLModels
import data_edit


def main():
    cleaner = data_edit.Data_Cleaner(['quantileTransform'])
    myMachine = MLModels(cleaner.X, cleaner.Y) 
    myMachine.splitTestTrain(0.3)
    list_best = myMachine.utilizeBest()
    return list_best

def GraphBest():
    best_list = main()
    

if __name__ == '__main__':
    main()
        
        
        
