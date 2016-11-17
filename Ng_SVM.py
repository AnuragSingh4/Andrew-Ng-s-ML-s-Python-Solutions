# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:17:02 2016

@author: anurag
"""

import pandas as pd
import numpy as np
from sklearn import svm
from scipy.io import loadmat 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


ex6data2=loadmat('ex6data1.mat')
X= ex6data['X']
y=ex6data1['y']
X1_new=[]
Y1_new=[]
X2_new=[]
Y2_new=[]

for i in range(y.size):
    if y[i]==1:
        X1_new.append(X[i,0])
        Y1_new.append(X[i,1])
    else:
        X2_new.append(X[i,0])
        Y2_new.append(X[i,1])
plt.scatter(X1_new,Y1_new,color='yellow')
plt.scatter(X2_new,Y2_new,color='blue')
plt.show()

linear_svc = svm.SVC(kernel='linear', C= 1)
linear_svc.fit(X,y)
linear_svc.support_vectors_
linear_svc.support_
linear_svc.n_support_
print(classification_report(y,linear_svc.predict(X)))


'''example two'''
ex6data2= loadmat('ex6data2.mat')
X2= ex6data2['X']
y2=ex6data2['y']
X21_new=[]
Y21_new=[]
X22_new=[]
Y22_new=[]

for i in range(y2.size):
    if y2[i]==1:
        X21_new.append(X2[i,0])
        Y21_new.append(X2[i,1])
    else:
        X22_new.append(X2[i,0])
        Y22_new.append(X2[i,1])
plt.scatter(X21_new,Y21_new,color='yellow')
plt.scatter(X22_new,Y22_new,color='blue')
linear_svc1 = svm.SVC(kernel='rbf', C= 100)
linear_svc1.fit(X2,y2)
linear_svc1.support_vectors_
linear_svc1.support_
no_sp=linear_svc1.n_support_
A=linear_svc1.support_
X1_plot=[]
Y1_plot=[]
X2_plot=[]
Y2_plot=[]
for i in A:
    if y2[i]==1:
        X1_plot.append(X2[i,0])
        Y1_plot.append(X2[i,1])
    else:
        X2_plot.append(X2[i,0])
        Y2_plot.append(X2[i,1])
plt.scatter(X1_plot,Y1_plot,color='red')
plt.scatter(X2_plot,Y2_plot,color='red')
plt.show()
print(classification_report(y2,linear_svc1.predict(X2)))
from sklearn.metrics import f1_score
print(f1_score(y2,linear_svc1.predict(X2)))




'''3rd data set'''

ex6data3=loadmat('ex6data3.mat')
X3= ex6data3['X']
y3=ex6data1['y']
X31_new=[]
Y31_new=[]
X32_new=[]
Y32_new=[]

for i in range(y3.size):
    if y3[i]==1:
        X31_new.append(X3[i,0])
        Y31_new.append(X3[i,1])
    else:
        X32_new.append(X3[i,0])
        Y32_new.append(X3[i,1])
plt.scatter(X31_new,Y31_new,color='yellow')
plt.scatter(X32_new,Y32_new,color='blue')
plt.show()


'''spam classifier '''




