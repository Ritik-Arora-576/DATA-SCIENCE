#!/usr/bin/env python
# coding: utf-8

# # LOAD DATASETS :

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


dfx=pd.read_csv('weightedX.csv')
dfy=pd.read_csv('weightedY.csv')


# In[13]:


X_values=dfx.values
Y_values=dfy.values


# In[14]:


plt.style.use('seaborn')
plt.scatter(X_values,Y_values)
plt.title('Wieghted Regression')
plt.xlabel('Wieghted x')
plt.ylabel('Wieghted y')
plt.show()


# # Let create a Wieghted matrix :
# 
# - Formula used :- exp(-sqr(x_query - x)/2*sqr(tau))

# In[16]:


def giveW(X,query_x,tau):
    total_points=X.shape[0]
    W=np.mat(np.eye(total_points))
    for i in range(total_points):
        exp_coff=np.dot(X[i]-query_x,(X[i]-query_x).T)/(2*tau*tau)
        W[i,i]=np.exp(-1*exp_coff)
    return W


# In[17]:


giveW(X_values,-1,1)


# # Predicted Value :

# In[26]:


def prediction(X_values,Y_values,query,tau):
    ones=np.ones((X_values.shape[0],1))
    X_values=np.concatenate((ones,X_values),axis=1)
    
    query=np.mat([1,query])
    
    W=giveW(X_values,query,tau)
    
    tetha=np.linalg.pinv(X_values.T*(W*X_values))*(X_values.T*(W*Y_values))
    pred=np.dot(query,tetha)
    return tetha,pred


# In[45]:


tetha,pred=prediction(X_values,Y_values,3,0.01)


# In[46]:


print(pred)


# **NOTE : 
# - On higher value of tau prediction curve follows linear regression and on low value of tau answer would be more accurate

# In[ ]:




