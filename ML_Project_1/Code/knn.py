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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# (Question 2)

# Put your funtions here
# ...


if __name__ == "__main__":
    
      

    
    """
        Question 2.1
    """
    X, y = make_unbalanced_dataset(3000,random_state=42)
    X_train, X_test = X[:1000,:], X[1000:,:]
    y_train, y_test = y[:1000], y[1000:]


    knc1= KNeighborsClassifier(n_neighbors = 1)
    knc5= KNeighborsClassifier(n_neighbors = 5)
    knc50= KNeighborsClassifier(n_neighbors = 50)
    knc100= KNeighborsClassifier(n_neighbors = 100)
    knc500= KNeighborsClassifier(n_neighbors = 500)

    knc1 = knc1.fit(X_train, y_train)
    knc5 = knc5.fit(X_train, y_train)
    knc50 = knc50.fit(X_train, y_train)
    knc100 = knc100.fit(X_train, y_train)
    knc500 = knc500.fit(X_train, y_train)    
    
    plot_boundary("figureknn1", knc1, X_test, y_test,0.1,"n_neighbors=1")  
    plot_boundary("figureknn5", knc5, X_test, y_test,0.1,"n_neighbors=5")    
    plot_boundary("figureknn50", knc50, X_test, y_test,0.1,"n_neighbors=50")    
    plot_boundary("figureknn100", knc100, X_test, y_test,0.1,"n_neighbors=100")    
    plot_boundary("figureknn500", knc500, X_test, y_test,0.1,"n_neighbors=500")    


    """
        Question 2.2
    """
    print("n_neighbors=1 / accuracy",cross_val_score(knc1, X, y, cv=10))
    print("n_neighbors=5 / accuracy",cross_val_score(knc5, X, y, cv=10))
    print("n_neighbors=50 / accuracy",cross_val_score(knc50, X, y, cv=10))
    print("n_neighbors=100 / accuracy",cross_val_score(knc100, X, y, cv=10))
    print("n_neighbors=500 / accuracy",cross_val_score(knc500, X, y, cv=10))


    print("n_neighbors=1 / mean accuracy",np.mean(cross_val_score(knc1, X, y, cv=10)))
    print("n_neighbors=5 / mean accuracy",np.mean(cross_val_score(knc5, X, y, cv=10)))
    print("n_neighbors=50 / mean accuracy",np.mean(cross_val_score(knc50, X, y, cv=10)))
    print("n_neighbors=100 / mean accuracy",np.mean(cross_val_score(knc100, X, y, cv=10)))
    print("n_neighbors=500 /  mean accuracy",np.mean(cross_val_score(knc500, X, y, cv=10)))


    
    """
    Question 2.3 and 2.4
    """

    X_test,y_test = make_unbalanced_dataset(500,random_state=42)
   
    seed=1
    #N=50
    mean_accuracy_50 = np.zeros(50)
    optimal = []
    for k in range(1,50):
        accuracy = np.zeros(10)
        mean_accuracy_50[0]=1
        for j in range(10):
            seed=seed+1
            X_train,y_train = make_unbalanced_dataset(50,random_state=seed)
            knn = KNeighborsClassifier(k)
            knn = knn.fit(X_train, y_train)
            accuracy[j]=knn.score(X_test,y_test)
            
            
        mean_accuracy_50[k]=np.mean(accuracy)
        mean_accuracy_50[0]=mean_accuracy_50[1]
    optimal.append(np.argmax(mean_accuracy_50))

        
    #N=150
    mean_accuracy_150 = np.zeros(150)

    for k in range(1,150):
        accuracy = np.zeros(10)
        mean_accuracy_150[0]=1
        for j in range(10):
            seed=seed+1
            X_train,y_train = make_unbalanced_dataset(150,random_state=seed)
            
            knn = KNeighborsClassifier(k)
            knn = knn.fit(X_train, y_train)
            accuracy[j]=knn.score(X_test,y_test)
            
            
        mean_accuracy_150[k]=np.mean(accuracy)
        mean_accuracy_150[0]=mean_accuracy_150[1]
    optimal.append(np.argmax(mean_accuracy_150))


    #N=250
    mean_accuracy_250 = np.zeros(250)

    for k in range(1,250):
        accuracy = np.zeros(10)
        mean_accuracy_250[0]=1
        for j in range(10):
            seed=seed+1
            X_train,y_train = make_unbalanced_dataset(250,random_state=seed)
            knn = KNeighborsClassifier(k)
            knn = knn.fit(X_train, y_train)
            accuracy[j]=knn.score(X_test,y_test)
            
            
        mean_accuracy_250[k]=np.mean(accuracy)
        mean_accuracy_250[0]=mean_accuracy_250[1]
    optimal.append(np.argmax(mean_accuracy_250))

    #N=350
    mean_accuracy_350 = np.zeros(350)

    for k in range(1,350):
        accuracy = np.zeros(10)
        mean_accuracy_350[0]=1
        for j in range(10):
            seed=seed+1
            X_train,y_train = make_unbalanced_dataset(350,random_state=seed)
            knn = KNeighborsClassifier(k)
            knn = knn.fit(X_train, y_train)
            accuracy[j]=knn.score(X_test,y_test)
            
            
        mean_accuracy_350[k]=np.mean(accuracy)
        mean_accuracy_350[0]=mean_accuracy_350[1]
    optimal.append(np.argmax(mean_accuracy_350))

    #N=450
    mean_accuracy_450 = np.zeros(450)
    optimal_450  = []

    for k in range(1,450):
        accuracy = np.zeros(10)
        mean_accuracy_450[0]=1
        for j in range(10):
            seed=seed+1
            X_train,y_train = make_unbalanced_dataset(450,random_state=seed)
            knn = KNeighborsClassifier(k)
            knn = knn.fit(X_train, y_train)
            accuracy[j]=knn.score(X_test,y_test)
            
            
        mean_accuracy_450[k]=np.mean(accuracy)
        mean_accuracy_450[0]=mean_accuracy_450[1]
    optimal.append(np.argmax(mean_accuracy_450))


    #N=500
    mean_accuracy_500 = np.zeros(500)
    optimal_500 = []

    for k in range(1,500):
        accuracy = np.zeros(10)
        mean_accuracy_500[0]=1
        for j in range(10):
            seed=seed+1
            X_train,y_train = make_unbalanced_dataset(500,random_state=seed)
            knn = KNeighborsClassifier(k)
            knn = knn.fit(X_train, y_train)
            accuracy[j]=knn.score(X_test,y_test)
           
            
        mean_accuracy_500[k]=np.mean(accuracy)
        mean_accuracy_500[0]=mean_accuracy_500[1]
    optimal.append(np.argmax(mean_accuracy_500))


    #Question 2.3.1
    plt.plot(np.array(range(1,51)) ,mean_accuracy_50, label='N=50')
    plt.plot(np.array(range(1,151)), mean_accuracy_150, label='N=150')
    plt.plot(np.array(range(1,251)), mean_accuracy_250, label='N=250')
    plt.plot(np.array(range(1,351)), mean_accuracy_350, label='N=350')
    plt.plot(np.array(range(1,451)), mean_accuracy_450, label='N=450')
    plt.plot(np.array(range(1,501)), mean_accuracy_500, label='N=500')
       
    plt.legend()
    plt.title("Mean test accuracy")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Evolution of mean test accuracies")
    plt.grid(True)
    plt.show()
  
    
    #Question 2.3.2
    """
    plt.xlabel("Test size")
    plt.ylabel("Number of neighbors")
    
    plt.plot([50,150,250,350,450,500] ,optimal)
    plt.show()
    """
    pass
