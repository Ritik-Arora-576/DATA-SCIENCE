#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[9]:


dfx=pd.read_csv('Linear_X_Train.csv')
dfy=pd.read_csv('Linear_Y_Train.csv')

x_values=dfx.values
y_values=dfy.values

mean=np.mean(x_values)
std=np.std(x_values)
x_values=(x_values-mean)/std


# In[34]:


def hypothetical(slope,y_intersept,x):
    return slope*x + y_intersept

def error(slope,y_intersept,x,y):
    total_error=0.0
    total_points=x.shape[0]
    for i in range(total_points):
        y_=hypothetical(slope,y_intersept,x[i])
        total_error+=(y_-y[i])**2
    return total_error/total_points

def gradient(slope,y_intersept,x,y):
    grad=np.zeros(2,)
    total_points=x.shape[0]
    for i in range(total_points):
        y_=hypothetical(slope,y_intersept,x[i])
        grad[0]+=(y_-y[i])*x[i]
        grad[1]+=(y_-y[i])
    return grad/total_points

def gradientDescent(x,y,learning_rate=0.1,max_steps=100):
    slope=0.0
    y_intersept=0.0
    error_data=[]
    line_data=[]
    for i in range(max_steps):
        grad=gradient(slope,y_intersept,x,y)
        slope=slope-learning_rate*grad[0]
        y_intersept=y_intersept-learning_rate*grad[1]
        e=error(slope,y_intersept,x,y)
        error_data.append(e)
        line_data.append([slope,y_intersept])
    return slope,y_intersept,error_data,line_data


# In[35]:


slope,y_intersept,error,line_data=gradientDescent(x_values,y_values)


# In[14]:


y_=hypothetical(slope,y_intersept,x_values)
plt.scatter(x_values,y_values)
plt.plot(x_values,y_,color='orange')
plt.show()


# In[27]:


a=np.arange(-40,40)
b=np.arange(-40,40)

a,b=np.meshgrid(a,b)

fig=plt.figure()
axes=fig.gca(projection='3d')
axes.contour(a,b,a**2+b**2,cmap='rainbow')
plt.show()
print(slope,y_intersept)


# In[49]:


slope1=np.arange(40,120)
y_intersept1=np.arange(-40,40)

slope1,y_intersept1=np.meshgrid(slope1,y_intersept1)
J=np.zeros(slope1.shape)
for i in range(slope1.shape[0]):
    for j in range(slope1.shape[1]):
        y_=slope1[i,j]*x_values + y_intersept1[i,j]
        J[i,j]=sum((y_values-y_)**2)/y_values.shape[0]
        


# In[50]:


fig=plt.figure()
axis=fig.gca(projection='3d')
axis.contour(slope1,y_intersept1,J,cmap='rainbow')
plt.show()


# In[52]:


plt.contour(slope1,y_intersept1,J,cmap='rainbow')


# In[54]:


line_data=np.array(line_data)
plt.style.use('seaborn')
plt.plot(line_data[:,0],color='green',label='slope')
plt.plot(line_data[:,1],color='orange',label='Y-Intersept')
plt.legend()
plt.show()


# In[60]:


fig=plt.figure()
axes=fig.gca(projection='3d')
axes.contour(slope1,y_intersept1,J,cmap='rainbow')
axes.scatter(line_data[:,0],line_data[:,1],error,cmap='rainbow')


# In[61]:


plt.contour(slope1,y_intersept1,J,cmap='rainbow')
plt.scatter(line_data[:,0],line_data[:,1])
plt.show()


# # Boston Housing Dataset :

# In[12]:


from sklearn.datasets import load_boston
boston=load_boston()
X=boston.data
y=boston.target
print(boston.feature_names)


# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfx=pd.DataFrame(X,columns=boston.feature_names)
dfy=pd.DataFrame(y)
dfx.describe()


# In[15]:


# normalisation of data
mean=np.mean(X,axis=0)
std=np.std(X,axis=0)
X=(X-mean)/std

dfx=pd.DataFrame(X,columns=boston.feature_names)


# In[19]:


dfx


# In[20]:


one=np.ones((X.shape[0],1))
X=np.concatenate((one,X),axis=1)


# In[23]:


print(X.shape,y.shape)


# In[31]:


plt.scatter(X[:,6],y)


# In[33]:


def hypothesis(tetha,x):
    y_=0.0
    n=tetha.shape[0]
    for i in range(n):
        y_+=tetha[i]*x[i]
    return y_

def error(X,y,tetha):
    error=0.0
    m,n=X.shape
    for i in range(m):
        y_=hypothesis(tetha,X[i,:])
        error+=(y_-y[i])**2
    return error/m

def gradient(X,y,tetha):
    m,n=X.shape
    grad=np.zeros((n,))
    for i in range(n):
        for j in range(m):
            y_=hypothesis(tetha,X[j])
            grad[i]+=(y_-y[j])*X[j,i]
    return grad/m
def gradient_descent(X,y,lr=0.1,max_steps=300):
    tetha=np.zeros((X.shape[1],))
    error_list=[]
    for i in range(max_steps):
        error_list.append(error(X,y,tetha))
        grad=gradient(X,y,tetha)
        for j in range(X.shape[1]):
            tetha[j]=tetha[j]-(lr*grad[j])
    return tetha,error_list


# In[39]:


import time
start=time.time()
tetha,error_list=gradient_descent(X,y)
end=time.time()


# In[40]:


plt.style.use('seaborn')
plt.plot(error_list)
plt.show()


# In[41]:


print(tetha)


# In[42]:


print(end-start)


# In[44]:


y_=[]
for i in range(X.shape[0]):
    y_pred=hypothesis(tetha,X[i])
    y_.append(y_pred)
y_=np.array(y_)

def r2_value(X,y,y_,tetha):
    y_avg=np.mean(y)
    return 1-(sum((y-y_)**2)/sum((y-y_avg)**2))

r2_value(X,y,y_,tetha)


# # Optimised way to solve multiple feature regression:
# - using np.dot() and np.sum()

# In[49]:


def hypothesis(X,tetha):
    return np.dot(X,tetha) # this returns a list of hypothetical values

def error(X,y,tetha):
    e=0.0
    m,n=X.shape
    y_=hypothesis(X,tetha)
    return np.sum((y-y_)**2)/m

def gradient(X,y,tetha):
    m,n=X.shape
    y_=hypothesis(X,tetha)
    grad=np.dot(X.T,(y_-y))
    return grad/m

def gradient_descent(X,y,max_steps=300,lr=0.1):
    m,n=X.shape
    error_list=[]
    tetha=np.zeros((n,))
    for i in range(max_steps):
        error_list.append(error(X,y,tetha))
        grad=gradient(X,y,tetha)
        tetha=tetha-(lr*grad)
    return tetha,error_list


# In[51]:


import time
start=time.time()
tetha,error_list=gradient_descent(X,y)
end=time.time()


# In[52]:


print(end-start)


# In[53]:


y_=[]
for i in range(X.shape[0]):
    y_pred=hypothesis(tetha,X[i])
    y_.append(y_pred)
y_=np.array(y_)

def r2_value(X,y,y_,tetha):
    y_avg=np.mean(y)
    return 1-(sum((y-y_)**2)/sum((y-y_avg)**2))

r2_value(X,y,y_,tetha)


# In[54]:


print(tetha)


# In[55]:


plt.plot(error_list)
plt.show()


# In[ ]:




