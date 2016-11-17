# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:07:39 2016

@author: anurag
"""

import pandas as pd
import matplotlib as mp
cols = ['popultion','profit']
ex1 = pd.read_csv('ex1data1.txt',names = cols)


X = ex1.iloc[:,0]
y = ex1.iloc[:,1]

d = pd.DataFrame({'1':1,'2':X})
theta = np.matrix([1,1])
new_d = np.matrix(d.values)
y = np.matrix(y.values)

def costfunc(new_d,y,theta):
    return sum(np.power((new_d*theta.T-y.T),2))/(2*new_d.shape[0])
def gradient(new_d, y, theta):
    return (((new_d*theta.T-y.T).T)*new_d)/(new_d.shape[0])
def gradient_descent(theta, alpha, new_d, y):
    theta = theta - alpha*(((new_d*theta.T-y.T).T)*new_d)/(new_d.shape[0])       
    return theta

cost_arr = np.zeros(1000)

for i in range(1,1000):
    theta = gradient_descent(theta, .01, new_d, y)
    cost_arr[i] = costfunc(new_d,y,theta)




