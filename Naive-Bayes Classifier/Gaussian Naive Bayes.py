#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.naive_bayes import GaussianNB


# In[3]:


gnb=GaussianNB()


# In[10]:


from sklearn.datasets import make_classification
X,y=make_classification(n_samples=200,n_features=2,n_informative=2,n_redundant=0,random_state=5)


# In[11]:


import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[12]:


gnb.fit(X,y)


# In[14]:


gnb.score(X,y)


# In[17]:


gnb.predict(X)


# In[18]:


print(y)


# - Bernaulli's Naive Bayes : It is used to classify our testing feature into 0 or 1 (Mail is a spam or not)
# 
# - Multinomial Naive Bayes : It is used when we have descrete data and deal with fraquency of occurance of specific feature in a training Data (Movie Rating ranging from 1 to 5 we have to consider frequency)
# 
# - Gaussian Naive Bayes : It is used for continous dataset (Irish dataset where we have consider petal length , petal width , sepal length , sepal width which have different values)

# In[ ]:




