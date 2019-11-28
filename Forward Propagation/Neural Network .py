#!/usr/bin/env python
# coding: utf-8

# ### Neural Network  
# 
# In this part of the exercise, you will implement a neural network to rec-
# ognize handwritten digits using the same training set as before. The neural
# network will be able to represent complex models that form non-linear hy-
# potheses. For this week, you will be using parameters from a neural network
# that we have already trained. Your goal is to implement the feedforward
# propagation algorithm to use our weights for prediction.

# In[101]:


# Import libraries for processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# To load matlab file import loadmat method from scipi.io
# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.io.loadmat.html

from scipy.io import loadmat


# In[102]:


mat=loadmat("ex3data1.mat")

X = mat["X"]
y = mat["y"]


# In[103]:


mat = loadmat('ex3weights.mat',appendmat=False)


# In[104]:


# theta1 - 25*401
# theta2 - 10*26

theta1 = mat['Theta1'] 
theta2 = mat['Theta2']


# #### Feedforward Propagation and Prediction
# 
# Now you will implement feedforward propagation for the neural network.
# 
# #### Octave Code
# 
# ```json
# function p = predict(Theta1, Theta2, X)
# 
# m = size(X, 1);
# num_labels = size(Theta2, 1);
# 
# % You need to return the following variables correctly 
# p = zeros(size(X, 1), 1);
# 
# a1 = [ones(m,1) X];
# 
# z2 = a1 * Theta1';
# a2 = sigmoid(z2);
# 
# a2 = [ones(size(a2,1),1) a2];
# 
# z3 = a2 * Theta2';
# a3 = sigmoid(z3);
# 
# 
# 
# [val, index] = max(a3,[],2);
# 
# p = index;
# ```

# In[105]:


def sigmoid(z):
    
    return 1/(1+np.exp(-z))


# In[106]:


def prediction(theta1,theta2,X):
    
    m = X.shape[0]
    
    interceptor = np.ones((m,1))
    
    X = np.c_[interceptor,X]
    
    z2 = X @ theta1.T
    a1 = sigmoid(z2)
    
    # Hidden Layer
    a1 = np.c_[interceptor,a1]
    
    z3 = a1 @ theta2.T
    a2 = sigmoid(z3)
    
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
    maxi = np.argmax(a2,axis = 1)+1
    # print(maxi)
    
    return maxi
    


# #### Formula to calculate Training Set Accuracy
# mean(double(pred == y)) * 100

# In[107]:


predicted_value = prediction(theta1,theta2,X)

# Calculate how many values are classified correctly by comparing it with actual value y.
print('Training Set Accuracy :',sum(predicted_value[:,np.newaxis]==y)[0]/5000*100,'%')

