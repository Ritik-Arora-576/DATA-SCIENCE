#!/usr/bin/env python
# coding: utf-8

# # DATA PREPRATION :-

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


mean=np.array([1,0.5])
cov=np.array([[1,0.1],[0.1,1.2]])

data=np.random.multivariate_normal(mean,cov,500)

mean1=np.array([4,5])
cov1=np.array([[1.2,0.1],[0.1,1.3]])

data1=np.random.multivariate_normal(mean1,cov1,500)

plt.style.use('seaborn')
plt.scatter(data[:,0],data[:,1],color='red',label='class 0')
plt.scatter(data1[:,0],data1[:,1],color='green',label='class 1')
plt.xlabel('x1 values')
plt.ylabel('x2 values')
plt.legend()
plt.show()


# In[3]:


# creating a datasets
dataset=np.zeros((1000,3))
dataset[:500,:-1]=data
dataset[500:,:-1]=data1
dataset[500:,-1]=1


# In[4]:


np.random.shuffle(dataset)


# In[5]:


# split datasets into train and test

split=int(0.8*dataset.shape[0])
train_x=dataset[:split,:-1]
train_y=dataset[:split,-1]
test_x=dataset[split:,:-1]
test_y=dataset[split:,-1]


# # NORMALISATION OF DATASETS :

# # visualise the testing data
# 
# plt.scatter(train_x[:,0],train_x[:,1],c=train_y)
# plt.show()

# In[6]:


# noramlise the traing and testing datasets
mean=np.mean(train_x,axis=0)
std=np.std(train_x,axis=0)

train_x=(train_x-mean)/std #normalise training data 
test_x=(test_x-mean)/std #normalise testing data

plt.scatter(train_x[:,0],train_x[:,1],c=train_y)
plt.show()


# # LOGISTIC REGRESSION IMPLEMENTATION :

# In[7]:


def sigmoid(value):
    return 1.0/(1.0+np.exp(-1*value))

def hypothesis(X,tetha):
    return sigmoid(np.dot(X,tetha))

def error(X,y,tetha):
    
    hi=hypothesis(X,tetha)
    e=(y*np.log(hi))+((1-y)*np.log(1-hi))
    return -1*np.mean(e)

def gradient(X,y,tetha):
    hi=hypothesis(X,tetha)
    grad=-1*np.dot(X.T,(y-hi))
    return grad/X.shape[0]

def gradient_descent(X,y,lr=0.5,max_steps=500):
    error_list=[]
    tetha=np.zeros((X.shape[1],1))
    for i in range(max_steps):
        e=error(X,y,tetha)
        error_list.append(e)
        grad=gradient(X,y,tetha)
        tetha=tetha-(lr*grad)
    return tetha,error_list


# In[8]:


one=np.ones((train_x.shape[0],1))
new_train_x=np.concatenate((one,train_x),axis=1)
train_y=train_y.reshape((-1,1))


# In[9]:


tetha,error_list=gradient_descent(new_train_x,train_y)


# In[10]:


error_list=np.array(error_list).reshape((-1,))


# In[11]:


plt.plot(error_list)
plt.show()


# In[12]:


plt.scatter(train_x[:,0],train_x[:,1],c=train_y.reshape((-1,)))
plt.show()


# In[13]:


a=np.arange(-3,4)

x2=-1*(tetha[0]+tetha[1]*a)/tetha[1]
plt.scatter(train_x[:,0],train_x[:,1],c=train_y.reshape((-1,)))
plt.plot(a,x2)
plt.show()


# # Classify the test datasets :

# In[14]:


one=np.ones((test_x.shape[0],1))
new_test_x=np.concatenate((one,test_x),axis=1)

test_y=test_y.reshape((-1,1))


# In[15]:


pred=hypothesis(new_test_x,tetha)
for i in range(pred.shape[0]):
    if pred[i]>0.5:
        pred[i]=1
    else:
        pred[i]=0
pred=pred.astype('int')


# In[16]:


def accuracy(pred,test_y):
    similar=0
    for i in range(pred.shape[0]):
        if pred[i]==test_y[i]:
            similar+=1
    return (similar/pred.shape[0])*100


# In[17]:


accuracy(pred,test_y)


# # Logistic Regression using sk-learn :

# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


model=LogisticRegression()
model.fit(train_x,train_y)


# In[20]:


tetha0=model.intercept_
tetha1=model.coef_
print(tetha0,tetha1)


# In[ ]:




