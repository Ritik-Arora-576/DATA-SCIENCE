#!/usr/bin/env python
# coding: utf-8

# # Data Preperation :

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('mushrooms.csv')
df


# In[10]:


# Our dataset is in alphabetical form but we want to convert it into numerical form
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds=df.apply(le.fit_transform)
ds


# In[11]:


ds=ds.values


# In[87]:


print(ds)
np.random.shuffle(ds)


# In[88]:


# Split Dataset into test and train data
split=int(0.8*ds.shape[0])
train_x=ds[:split,1:]
test_x=ds[split:,1:]
train_y=ds[:split,0]
test_y=ds[split:,0]


# # Naive Classifier :

# In[89]:


a=np.array([1,1,1,0,0,0,1,0,0,1,0])
float(np.sum(a==1))


# In[90]:


def prior_probablity(y_train,label):
    total=y_train.shape[0]
    favourable=np.sum(y_train==label)
    return favourable/float(total)


# In[91]:


def cond_probablity(x_train,y_train,label,feature_col,feature_val):
    x_filtered=x_train[y_train==label]  # it is used to shrink the values of dataset whose class is label
    numerator=np.sum(x_filtered[:,feature_col]==feature_val)
    denominator=np.sum(y_train==label)
    return numerator/float(denominator)


# In[92]:


def prediction(x_train,y_train,x_test):
    uni=np.unique(y_train)
    n_features=x_train.shape[1]
    pred=[]
    for label in uni:
        likelihood=1.0
        for i in range(n_features):
            c=cond_probablity(x_train,y_train,label,i,x_test[i])
            likelihood*=c
        
        prior=prior_probablity(y_train,label)
        post=prior*likelihood
        pred.append(post)
        
    pred=np.array(pred)
    index=np.argmax(pred)
    return uni[index]


# In[93]:


print(prediction(train_x,train_y,test_x[764]))
print(test_y[764])


# In[94]:


def score(x_train,y_train,x_test,y_test):
    pred=[]
    for i in range(x_test.shape[0]):
        pred.append(prediction(x_train,y_train,x_test[i]))
    pred=np.array(pred)
    return np.sum(pred==y_test)/x_test.shape[0]


# In[95]:


score(train_x,train_y,test_x,test_y)


# In[ ]:




