#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# In[67]:


df=pd.read_csv('train.csv')

x=df.values[:,:-1]
y=df.values[:,-1].reshape(-1,1)


# In[22]:


def weighted_matrix(x,tau,query):
    w=np.mat(np.eye(x.shape[0]))
    for i in range(x.shape[0]):
        exp_coff=np.dot((x[i]-query),(x[i]-query).T)/(2*tau*tau)
        w[i,i]=np.exp(-1*exp_coff)
    return w


# In[57]:


def prediction(x,y,tau,query):
    one=np.ones((x.shape[0],1))
    x=np.concatenate((one,x),axis=1)
    
    query=list(query)
    query.insert(0,1)
    query=np.array(query)
    w=weighted_matrix(x,tau,query)
    tetha=np.linalg.pinv(x.T*(w*x))*(x.T*(w*y))
    return np.dot(query,tetha)


# In[58]:


df=pd.read_csv('test.csv')

testing_data=df.values


# In[64]:


values=[]
for i in range(testing_data.shape[0]):
    pred=prediction(x,y,1,testing_data[i])
    values.append(float(pred))


# In[100]:


values=np.array(values).reshape(-1,1)
id=np.arange(400).reshape(-1,1)
data=np.concatenate((id,values),axis=1)


# In[106]:


df=pd.DataFrame(values,columns=['target'])


# In[107]:


df.to_csv('Air pollution.csv')


# In[108]:


get_ipython().run_line_magic('pinfo', 'df.to_csv')


# In[ ]:




