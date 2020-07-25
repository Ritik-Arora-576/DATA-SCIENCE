#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')

# First column represent serial number thats why we have to eliminate it
x_val=dfx.values[:,1:]
y_val=dfy.values[:,1:].reshape(-1,)


# In[3]:


# plot graph between x and y cordinates present in x_val and seperate them by using y_val 
plt.figure(figsize=(9,7))
plt.scatter(x_val[:,0],x_val[:,1],c=y_val,marker='*')
plt.show()


# In[4]:


plt.figure(figsize=(9,7))
query_x=np.array([2,3])
plt.scatter(query_x[0],query_x[1],color='red',marker='o')
plt.scatter(x_val[:,0],x_val[:,1],c=y_val,marker='*')
plt.show()


# In[5]:


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


# In[43]:


df=pd.read_csv('mnist_train.csv')
label=df.values[:,0]
data=df.values[:,1:]
split=int(0.8*label.shape[0])
train_label=df.values[ :split,0]
test_label=df.values[split:,0]
train_data=df.values[:split,1:]
test_data=df.values[split:,1:]
print(train_data.shape)


# In[49]:


def draw_image(data):
    plt.figure(figsize=(2,2))
    plt.axis('off')
    plt.imshow(data.reshape(28,28),cmap='gray')
    plt.show()
    
draw_image(test_data[989])
#print(train_label[67])


# In[48]:


prediction=knn(test_data[989],train_data,train_label)
print(prediction)


# In[ ]:




