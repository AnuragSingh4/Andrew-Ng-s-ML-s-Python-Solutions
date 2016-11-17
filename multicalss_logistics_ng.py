# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:40:50 2016

@author: anurag
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat 

theta = np.zeros(shape = (400,1))

X=ex3data1['X']
y=ex3data1['y']
X= np.matrix(X)
y=np.matrix(y)
def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))

def costfunc(X,y,theta,lamb):
    return((y.T)*np.log(sigmoid(X.dot(theta)))+((1-y).T)*np.log(sigmoid(1-X.dot(theta)))) /(2*X.shape[0]) + lamb*(theta*theta.T)

def gradientdescent(X, y, theta, lamb):
    return theta - (X.T)*(sigmoid(X.dot(theta))-y) +lamb*np.sum(theta)
    
J=[]
for i in range(10):
    theta = gradientdescent(X, y, theta, 0)
    J.append(costfunc(X,y,theta,0))
    
from sklearn import linear_model, decomposition, datasets
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X,y)
sum((y_predict.T-y)==0)




