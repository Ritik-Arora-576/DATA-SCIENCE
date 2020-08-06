#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


# In[15]:


X,y=make_regression(n_samples=10000,n_features=20,n_informative=20,noise=10,random_state=1)


# normalisation of datasets 
mean=np.mean(X,axis=0)
std=np.std(X,axis=0)
X=(X-mean)/std

x=X


# In[16]:


one=np.ones((X.shape[0],1))
X=np.concatenate((one,x),axis=1)
print(X.shape)


# In[20]:


dfx=pd.DataFrame(X)
dfx


# In[26]:


def hypothesis(X,tetha):
    return np.dot(X,tetha)
def error(X,y,tetha):
    y_=hypothesis(X,tetha)
    return sum((y_-y)**2)/X.shape[0]
def gradient(X,y,tetha):
    y_=hypothesis(X,tetha)
    return np.dot(X.T,(y_-y))/X.shape[0]
def gradient_descent(X,y,max_steps=300,lr=0.01):
    tetha=np.zeros((X.shape[1],))
    error_list=[]
    for i in range(max_steps):
        grad=gradient(X,y,tetha)
        error_list.append(error(X,y,tetha))
        tetha=tetha-(lr*grad)
    return tetha,error_list


# In[27]:


tetha,error_list=gradient_descent(X,y)


# In[28]:


plt.plot(error_list)


# # Mini Batch Gradient Descent  (OPTIMIZATION TECHNIQUE):
# 
# - Less time consuming
# - Noisy process avoid local minima
# 
# * Here we break whole dataset into batches of certain batch size and do regular updates *
# 
# - It is not more exact but close to exact value

# In[31]:


def mini_batch_gradient(X,y,max_steps=50,lr=0.01,batch_size=200):
    tetha=np.zeros((X.shape[1],))
    data=np.concatenate((X,y.reshape((-1,1))),axis=1)
    error_list=[]
    for i in range(max_steps):
        np.random.shuffle(data)
        n_batch=X.shape[0]//batch_size
        error_list.append(error(X,y,tetha))
        for j in range(n_batch):
            batch_data=data[j*batch_size:(j+1)*batch_size,:]
            batch_X=batch_data[:,:-1]
            batch_y=batch_data[:,-1]
            grad=gradient(batch_X,batch_y,tetha)
            tetha=tetha-(lr*grad)
    return tetha,error_list


# In[32]:


tetha,error_list=mini_batch_gradient(X,y)


# In[33]:


plt.plot(error_list)


# In[ ]:




