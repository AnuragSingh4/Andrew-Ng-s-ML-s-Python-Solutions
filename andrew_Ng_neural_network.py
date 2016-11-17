# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:50:18 2016

@author: anurag
"""
import pandas as pd
import numpy as np
from scipy.io import loadmat 
from sklearn.preprocessing import OneHotEncoder 
ex3data3=loadmat('ex3data1.mat')
X=ex3data1['X']
y=ex3data1['y']
X= np.matrix(X)
y=np.matrix(y)


'''feed forward method'''
ex4weights = loadmat('ex4weights.mat')
Theta1= ex4weights['Theta1']
Theta2=ex4weights['Theta2']

def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))

def sigmoid_grad(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

def forward_prop(Theta1,Theta2,X):
    a1= X
    a1 = np.insert(a1,0,values = np.ones(a1.shape[0]),axis=1)
    z2 = a1*Theta1.T
    a2 = sigmoid(z2)
    a2= np.insert(a2, 0 ,values = np.ones(a2.shape[0]),axis = 1)
    z3= a2*Theta2.T
    a3=sigmoid(z3)
    h=a3
    return h
    

pred=[]
for i in range(5000):
    pred.append(np.argmax(a3[i])+1)
    
predicted = np.matrix(pred).T

error = 5000- sum((predicted-y)==0)


'''back propagation'''
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y) 
y_onehot.shape
input_size= 400
hidden_size=25
num_labels=10
learning_rate=1
params= np.random.random(size=(input_size+1)*hidden_size +(hidden_size+1)*num_labels)

def costfunc(params, input_size, hidden_size, num_labels,X,y,learning_rate):
    y_onehot = encoder.fit_transform(y) 
    m = X.shape[0]
    theta1= np.matrix(np.reshape(params[:(input_size+1)*hidden_size],(hidden_size,input_size+1)))
    theta2=np.matrix(np.reshape(params[(input_size+1)*hidden_size:],(num_labels,hidden_size+1)))
    a1= X
    a1 = np.insert(a1,0,values = np.ones(a1.shape[0]),axis=1)
    z2 = a1*theta1.T
    a2 = sigmoid(z2)
    a2= np.insert(a2, 0 ,values = np.ones(a2.shape[0]),axis = 1)
    z3= a2*theta2.T
    a3=sigmoid(z3)
    h= a3
    J = (-np.sum(np.multiply(y_onehot,np.log(h)))-np.sum(np.multiply((1-y_onehot),np.log(1-h))))/m
    J_wr = J + (learning_rate/(2*m))*np.sum(np.multiply(theta1[:,1:],theta1[:,1:]))+(learning_rate/(2*m))*np.sum(np.multiply(theta2[:,1:],theta2[:,1:]))
    d3t = h-y
    d2t=np.multiply(d3t*theta2,sigmoid_grad(z2))
    delta2= d3t.T*a2
    delta1= d2t.T*a1
    delta1= delta1[1:,:]/m + theta1*learning_rate/m
    delta2=delta2[:,1:]/m + theta2[:,1:]*learning_rate/m
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J, grad

from scipy.optimize import minimize
fmin = minimize(fun=costfunc, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),  
                method='TNC', jac=True, options={'maxiter': 250})



 





