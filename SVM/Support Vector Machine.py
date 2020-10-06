#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine :

# In[1]:


from sklearn.datasets import make_classification
import numpy as np


# In[2]:


X,y=make_classification(n_samples=400,n_classes=2,n_features=2,n_informative=2,n_redundant=0,random_state=3,n_clusters_per_class=1)
y[y==0]=-1


# In[3]:


from matplotlib import pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[4]:


class SVM:
    def __init__(self,C=1.00):
        self.C=C
        self.W=0
        self.b=0
        
    def hingeloss(self,W,b,X,y):
        loss=0.0
        loss+=0.5*np.dot(W,W.T)
        
        m=X.shape[0]
        for i in range(m):
            ti=y[i]*(np.dot(W,X[i].T)+b)
            loss+=self.C*max(0,1-ti)
        return loss[0][0]
    
    def fit(self,X,y,batch_size=100,learning_rate=0.001,maxitr=50):
        
        no_of_features=X.shape[1]
        no_of_samples=X.shape[0]
        
        n=learning_rate
        c=self.C
        
        #initialize the model parameters
        W=np.zeros((1,no_of_features))
        b=0
        
        losses=[]
        
        for i in range(maxitr):
            losses.append(self.hingeloss(W,b,X,y))
            ids=np.arange(no_of_samples)
            np.random.shuffle(ids)
            for batch_start in range(0,no_of_samples,batch_size):
                gradw=0.0
                gradb=0.0
                
                for j in range(batch_start,batch_start+batch_size):
                    if j<no_of_samples:
                        idx=ids[j]
                        ti=y[idx]*(np.dot(W,X[idx].T)+b)
                        if ti>=1:
                            gradw=0
                            gradb=0
                        else:
                            gradw+=c*y[idx]*X[idx]
                            gradb+=c*y[idx]
                W=W- n*W + n*gradw
                b=b - n*gradb
        self.W=W
        self.b=b
        return W,b,losses
                        
        


# In[5]:


X=np.array(X)


# In[6]:


mySVM=SVM(30)


# In[7]:


W,b,loss=mySVM.fit(X,y)


# In[8]:


plt.plot(loss)


# In[9]:


def print_hyperplane(X,y):
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0],X[:,1],c=y)
    x1=np.arange(-2,4)
    w1=W[0,0]
    w2=W[0,1]
    x2=-(w1*x1 + b)/w2
    xp=-(w1*x1 + b+1)/w2
    xn=-(w1*x1 + b-1)/w2
    plt.plot(x1,x2,label='Hyperplane')
    plt.plot(x1,xn,color='red',label='Negetive hyperplane')
    plt.plot(x1,xp,color='green',label='Positive hyperplane')
    plt.legend()
    plt.show()
    return


# In[10]:


print_hyperplane(X,y)


# # Classify Non-Linear Datasets :

# - To classify non-linear datasets we have to convert dataset into higher dimensions

# In[11]:


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np


# In[12]:


X,y=make_circles(n_samples=500,noise=0.02)
X=np.array(X)


# In[13]:


plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[14]:


# convert 2-D datset into 3-D dataset

def convert_into_higher_dimension(X):
    x1=X[:,0]
    x2=X[:,1]
    x3=x1**2 + x2**2
    
    X_=np.zeros((X.shape[0],X.shape[1]+1))
    X_[:,:-1]=X
    X_[:,-1]=x3
    
    return X_


# In[15]:


X_=convert_into_higher_dimension(X)


# In[16]:


X_


# In[17]:


# plot data in 3-D surface

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_[:,0],X_[:,1],X_[:,2],c=y)
plt.show()


# In[18]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_,y)


# In[19]:


lr.coef_


# In[20]:


lr.intercept_


# In[21]:


a=lr.coef_[0,0]
b=lr.coef_[0,1]
c=lr.coef_[0,2]
d=lr.intercept_[0]


# In[ ]:





# In[22]:


xx,yy=np.meshgrid(range(-2,2),range(-2,2))
print(xx)
print(yy)


# In[23]:


# ax+ by+ cz+ d =0
z=-(a*xx + b*yy +d)/c
print(z)


# In[24]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_[:,0],X_[:,1],X_[:,2],c=y)
ax.plot_surface(xx, yy, z,alpha=0.2)
plt.show()


# # Kernel Trick :

# In[27]:


# when kernel is rbf

from sklearn import svm
svc=svm.SVC(kernel='rbf')
svc.fit(X,y)
svc.score(X,y)


# In[28]:


# when kernel is linear

svc=svm.SVC(kernel='linear')
svc.fit(X,y)
svc.score(X,y)


# In[29]:


# when kernel is polynomial

svc=svm.SVC(kernel='poly')
svc.fit(X,y)
svc.score(X,y)


# In[32]:


# custom kernel
def custom_kernel(X1,X2):
    return np.square(np.dot(X1,X2.T))

svc=svm.SVC(kernel=custom_kernel)
svc.fit(X,y)
svc.score(X,y)


# # Classification using MNIST datasets :

# In[1]:


# Data Prepration

from sklearn.datasets import load_digits


# In[2]:


digits=load_digits()
X=digits.data
y=digits.target


# In[13]:


# predict our model using logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lr=LogisticRegression()

lr.fit(X,y)
cross_val_score(lr,X,y,cv=5).mean()


# ### By logistic regression we got 91.3% accuracy 

# In[28]:


# predict our model using svm kernels :

from sklearn import svm

svc=svm.SVC()
svc.fit(X,y)
cross_val_score(svc,X,y,scoring='accuracy',cv=5).mean()


# ### By SVM we got 96.3% accuracy

# # Determine the best kernel and value of c such that accuracy will be maximum

# In[37]:


params=[
    {
        'kernel':['linear','rbf','sigmoid','poly'],
        'C':[0.1,0.2,0.5,1.0,2.0,5.0]
    }
]


# In[38]:


from sklearn.model_selection import GridSearchCV


# In[39]:


gs=GridSearchCV(estimator=svc,param_grid=params,scoring='accuracy',cv=5,n_jobs=8)


# In[32]:


# determine the number of CPU support by our machine
import multiprocessing
n_cpu=multiprocessing.cpu_count()


# In[33]:


n_cpu


# In[40]:


gs.fit(X,y)


# In[41]:


gs.best_estimator_


# In[42]:


gs.best_score_


# ### we got highest accuracy of 97.3% on putting the value of C=5.0 and kernel would be rbf

# In[ ]:




