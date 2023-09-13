"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import plot_tree




if __name__ == "__main__":
    
    X, y = make_unbalanced_dataset(3000,random_state=42)

    # Divide the dataset
    # first 1000 as training set
    X_train, X_test = X[:1000,:], X[1000:,:]
    y_train, y_test = y[:1000], y[1000:]

    
    
    clf2 = DecisionTreeClassifier(min_samples_split=2,random_state=42)
    clf8 = DecisionTreeClassifier(min_samples_split=8,random_state=42)
    clf32 = DecisionTreeClassifier(min_samples_split=32,random_state=42)
    clf64 = DecisionTreeClassifier(min_samples_split=64,random_state=42)
    clf128 = DecisionTreeClassifier(min_samples_split=128,random_state=42)
    clf500 = DecisionTreeClassifier(min_samples_split=500,random_state=42)


    """
      Fit the model
    """ 
    clf2 = clf2.fit(X_train, y_train)
    clf8 = clf8.fit(X_train, y_train)
    clf32 = clf32.fit(X_train, y_train)
    clf64 = clf64.fit(X_train, y_train)
    clf128 = clf128.fit(X_train, y_train)
    clf500 = clf500.fit(X_train, y_train)
   
    """
       Plot boundary for each hyper-parameter
    """
    plot_boundary("figure1", clf2, X_test, y_test,0.1,"min_samples_split=2")    
    plot_boundary("figure2", clf8, X_test, y_test,0.1,"min_samples_split=8")    
    plot_boundary("figure3", clf32, X_test, y_test,0.1,"min_samples_split=32")    
    plot_boundary("figure4", clf64, X_test, y_test,0.1,"min_samples_split=64")    
    plot_boundary("figure5", clf128, X_test, y_test,0.1,"min_samples_split=128")    
    plot_boundary("figure6", clf500, X_test, y_test,0.1,"min_samples_split=500")    



    
    """
        Test accuracy
    """
    min_samples_split_array=[2,8,32,64,128,500]
  
    for k in range(len(min_samples_split_array)):
        accuracy=np.empty(10)
        print("Hyper parameter=",min_samples_split_array[k])
        for j in range(10):
            X,y = make_unbalanced_dataset(3000,random_state=j)
            X_train, X_test = X[:1000,:], X[1000:,:]
            y_train, y_test = y[:1000], y[1000:]
            clf = DecisionTreeClassifier(min_samples_split=min_samples_split_array[k])
            clf = clf.fit(X_train, y_train)
        
            accuracy[j]=clf.score(X_test,y_test)
        print("Accuracy=",accuracy)
        print("Mean=",np.mean(accuracy))
        print("Std=",np.std(accuracy))


                
       




        
        
        
   
    pass
