# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:29:18 2018

@author: dillon
"""

import scipy.io as sio
from sklearn import svm
import numpy as np
import h5py
import pandas as pd

traindata = sio.loadmat('spamTrain.mat')
X = traindata['X']
y = traindata['y']
##Validation dataset is saved in a different format
##So the process to load it is slightly different
f = h5py.File('spamTest2.mat','r') 
Xtest = np.array(f.get('Xtest'))
Xtest = Xtest.T
ytest = np.array(f.get('ytest'))
ytest = ytest.T

##Train SVM with linear kernel
##Use for loop to determine best value for regularization parameter C
C = [0.01,0.03,0.1,0.3,1,3,10]
trainingAccuracy = np.empty([len(C)])
testingAccuracy = np.empty([len(C)])
f1 = np.empty([len(C)])
for i in range(0,len(C)):
    clf = svm.SVC(C=C[i],kernel='linear')
    clf.fit(X, y.ravel())

    trainPredict = clf.predict(X)
    trainingAccuracy[i] = np.mean(trainPredict==y.ravel())*100
    
    testPredict = clf.predict(Xtest)
    testingAccuracy[i] = np.mean(testPredict==ytest.ravel())*100
    
    #f1 score
    tp = (np.logical_and(testPredict[:] == 1,ytest.ravel()[:] == 1))
    p = ytest.ravel() == 1
    recall = sum(tp)/sum(p)
    fp = (np.logical_and(testPredict[:] == 1,ytest.ravel()[:] == 0))
    precision = sum(tp)/(sum(tp)+sum(fp))
    f1[i] = (2*(precision*recall))/(precision+recall)
    
print(testingAccuracy)
print(trainingAccuracy)
print(f1)