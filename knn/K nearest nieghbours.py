#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[15]:


dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')

# First column represent serial number thats why we have to eliminate it
x_val=dfx.values[:,1:]
y_val=dfy.values[:,1:].reshape(-1,)


# In[26]:


# plot graph between x and y cordinates present in x_val and seperate them by using y_val 
plt.figure(figsize=(9,7))
plt.scatter(x_val[:,0],x_val[:,1],c=y_val,marker='*')
plt.show()


# In[29]:


plt.figure(figsize=(9,7))
query_x=np.array([2,3])
plt.scatter(query_x[0],query_x[1],color='red',marker='o')
plt.scatter(x_val[:,0],x_val[:,1],c=y_val,marker='*')
plt.show()


# In[60]:


def distance(x1,x2):
    return (np.sqrt(sum((x1-x2)**2)))

def knn(query_x,x_val,y_val,m=5):
    val=[]
    total_points=x_val.shape[0]
    for i in range(total_points):
        dist=distance(x_val[i],query_x)
        val.append((dist,y_val[i]))
        
    val=sorted(val)
    val=val[:m]
    val=np.array(val)
    uni=np.unique(val[:,1],return_counts=True)
    index=uni[1].argmax()
    return uni[0][index]
    
knn([3.0,0.0],x_val,y_val)


# In[ ]:




