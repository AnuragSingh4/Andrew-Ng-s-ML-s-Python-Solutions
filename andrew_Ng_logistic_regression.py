# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:28:02 2016

@author: anurag
"""

import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn import linear_model, decomposition, datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2)
def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))
ex2data1 = pd.read_csv('ex2data1.txt',names = ['exam1score','exam2score','admitted'])
no_cols = ex2data1.shape[1]
X  = ex2data1.iloc[:,0:no_cols-1]
y = ex2data1.iloc[:,no_cols-1:no_cols]
X = np.c_[np.ones(X.shape[0]),X]
X = preprocessing.scale(X)
theta = np.matrix([[0],[0],[0]])
y= np.matrix(y)

def costfunc(X,y,theta,lamb):
    cost1 = (y.T)*np.log(sigmoid(X.dot(theta))) 
    cost2 = ((1-y).T)*np.log(sigmoid(1-X.dot(theta)))
    J = (cost1+cost2)/(2*X.shape[0])
    return (cost1+cost2)/(2*X.shape[0])

def gradientdescent(X, y, theta, alpha):
    return theta - (X.T)*(sigmoid(X*theta)-y) 
cost1=[]
cost2=[]
J=[]
'''82% accuracy'''
for i in range(20000):
    theta = gradientdescent(X, y, theta, 0.01)
    J.append(costfunc(X,y,theta))

'''using scikit learn//89% accuracy'''    
from sklearn import linear_model, decomposition, datasets
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X,y)

probability = sigmoid(X.dot(theta2))

def predict(theta, X):  
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]



    

