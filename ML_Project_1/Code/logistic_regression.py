"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import exp
from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=5000, learning_rate=0.01):
        self.n_iter = n_iter
        self.learning_rate = learning_rate


    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")
    
        N=X.shape[0] 
        theta=np.array([0,0,0])
        for t in range(0,self.n_iter):
            gradient= np.zeros((1,3))
            for i in range(N): 
                Xprime=np.array([1,X[i][0],X[i][1]])
                tmp = theta*Xprime
                tmp=np.exp(-tmp)
                tmp=1/(1+tmp)
                tmp = tmp - y[i]
                tmp = tmp * Xprime
                gradient = gradient + tmp
            gradient=(1/N)*gradient
            theta=theta-self.learning_rate*gradient
        self.theta=theta
        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        
        y= np.zeros(X.shape[0])
        
        tmp=self.predict_proba(X)
        
        for i in range(X.shape[0]):
            y[i] = np.argmax(tmp[i])
        return y            
        

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
                
        p = np.zeros((X.shape[0],2))
        N=X.shape[0]
        
        for i in range(N):
            z =  self.theta[0][0]+self.theta[0][1]*X[i][0]+self.theta[0][2]*X[i][1]
            tmp = np.exp(-z)
            tmp = 1+tmp
            tmp = 1/tmp
            p[i][1]=tmp
            p[i][0]=1-tmp
        return p
        
        
if __name__ == "__main__":
    
    
    X, y = make_unbalanced_dataset(3000,random_state=42)

    X_train, X_test = X[:1000,:], X[1000:,:]
    y_train, y_test = y[:1000], y[1000:]
    lrc=LogisticRegressionClassifier()
    lrc=lrc.fit(X_train,y_train)
    lrc.predict_proba(X_test)
    lrc.predict(X_test)
    
    #Question 3.5
    plot_boundary("figurelogistic_regression_iter=500", lrc, X_test, y_test,0.1,"n_iterations=500")    

    
    #Question 3.6
    accuracy=np.empty(5)
    for j in range(5):
        X,y = make_unbalanced_dataset(3000,random_state=j+10)
        X_train, X_test = X[:1000,:], X[1000:,:]
        y_train, y_test = y[:1000], y[1000:]
        lrc = LogisticRegressionClassifier()
        lrc = lrc.fit(X_train, y_train)
    
        accuracy[j]=lrc.score(X_test,y_test)
    print("Accuracy=",accuracy)
    print("Mean=",np.mean(accuracy))
    print("Std=",np.std(accuracy))

    pass
