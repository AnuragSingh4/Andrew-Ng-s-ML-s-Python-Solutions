# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:14:05 2016

@author: anurag
"""

import pandas as pd
import matplotlib as mp
from sklearn import preprocessing
cols = ['size','bedroom','price']
ex1 = pd.read_csv('ex1data2.txt',names = cols)
ex1.insert(0,'Ones',1)
no_cols= ex1.shape[1]

X = ex1.iloc[:,0:no_cols-1]
y = ex1.iloc[:,no_cols-1:no_cols]
X_scaled = preprocessing.scale(X)
y_scaled = preprocessing.scale(y)

theta = np.matrix([1,1,1])
new_d = X_scaled
y = y_scaled

def costfunc(new_d,y,theta):
    return sum(np.power((new_d*theta.T-y),2))/(2*new_d.shape[0])
def gradient(new_d, y, theta):
    return (((new_d*theta.T-y).T)*new_d)/(new_d.shape[0])
def gradient_descent(theta, alpha, new_d, y):
    theta = theta - alpha*(((new_d*theta.T-y).T)*new_d)/(new_d.shape[0])       
    return theta

cost_arr = np.zeros(1000)

for i in range(1,1000):
    theta = gradient_descent(theta, .01, new_d, y)
    cost_arr[i] = costfunc(new_d,y,theta)