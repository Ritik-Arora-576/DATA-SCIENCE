#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


dfx=pd.read_csv('Logistic_X_Train.csv')
dfy=pd.read_csv('Logistic_Y_Train.csv')

train_x=dfx.values
train_y=dfy.values


# In[9]:


#normalisation of data
mean=train_x.mean(axis=0)
std=train_x.std(axis=0)
train_x=(train_x-mean)/std


# In[42]:


def sigmoid(value):
    return 1.0/(1.0+np.exp(-1*value))

def hypothesis(X,tetha):
    return sigmoid(np.dot(X,tetha))

def error(X,y,tetha):
    hi=hypothesis(X,tetha)
    e=y*(np.log(hi))+((1-y)*np.log(1-hi))
    return -1*np.mean(e)

def gradient(X,y,tetha):
    hi=hypothesis(X,tetha)
    return -1*np.dot(X.T,(y-hi))/X.shape[0]

def gradient_descent(X,y,max_steps=500,lr=1.0):
    tetha=np.zeros((X.shape[1],1))
    error_list=[]
    for i in range(max_steps):
        error_list.append(error(X,y,tetha))
        grad=gradient(X,y,tetha)
        tetha=tetha-(lr*grad)
    return error_list,tetha


# In[43]:


one=np.ones((train_x.shape[0],1))
new_train_x=np.concatenate((one,train_x),axis=1)

error_list,tetha=gradient_descent(new_train_x,train_y)


# In[47]:


df=pd.read_csv('Logistic_X_test.csv')

test_x=df.values
test_x=(test_x-mean)/std
one=np.ones((test_x.shape[0],1))
new_test_x=np.concatenate((one,test_x),axis=1)


# In[48]:


pred=[]
for i in range(new_test_x.shape[0]):
    prediction=hypothesis(new_test_x[i],tetha)
    if prediction>=0.5:
        pred.append(1)
    else:
        pred.append(0)


# In[49]:


df=pd.DataFrame(pred,columns=['label'])
df.to_csv('Chemicals.csv',index=False)


# In[ ]:




