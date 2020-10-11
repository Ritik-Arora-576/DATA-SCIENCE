#!/usr/bin/env python
# coding: utf-8

# ### Dataset Prepration :

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[2]:


X,y=make_blobs(n_samples=500,n_features=2,centers=5,random_state=3)


# In[3]:


plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[4]:


# thus it is unsupervised learning , so we do not suppose to assign color to the datapoints , we have to determine by own
plt.scatter(X[:,0],X[:,1])
plt.grid(True)
plt.show()


# ### Give Random position to cluster :

# In[5]:


k=5 # number of clusters

colors=['green','yellow','blue','red','orange']

clusters={}

for i in range(k):
    
    # we have to give random points related to that cluster
    # we have to assign points related to that cluster
    # we have to assign clusters related to that cluster
    
    center=10*(2*np.random.random((X.shape[1],))-1) # gives random point between -10 and 10
    
    points=[]
    
    clusters[i]={ 'color':colors[i],
                 'points':points,
                 'center':center
    }


# In[6]:


clusters


# In[7]:


def distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


# In[8]:


a=np.array([1,0])
b=np.array([0,1])
distance(a,b)


# In[9]:


# classify the points for every cluster

def divide_points_by_clusters(clusters): # E-step
    for i in range(X.shape[0]):
        dist=[]
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        
        idx=np.argmin(dist)
        clusters[idx]['points'].append(X[i])


# In[10]:


def plot_clusters(clusters):
    for i in range(k):
        plt.scatter(clusters[i]['center'][0],clusters[i]['center'][1],c='black',marker='X')
        try:
            plt.scatter(np.array(clusters[i]['points'])[:,0],np.array(clusters[i]['points'])[:,1],c=clusters[i]['color'])
        except:
            pass
    plt.show()
    return


# In[11]:


divide_points_by_clusters(clusters)
plot_clusters(clusters)


# In[12]:


def update_centers(clusters): # M-step
    for i in range(k):
        pts=np.array(clusters[i]['points'])
        if pts.shape[0]>0:
            clusters[i]['center']=np.mean(pts,axis=0)
        clusters[i]['points']=[]
    return


# In[13]:


for i in range(10):
    divide_points_by_clusters(clusters)
    plot_clusters(clusters)
    update_centers(clusters)


# # K-means using Sk-learn library :
# 
# - It is used to initialize centers in such a way that the accuracy would be maximize and loss should be minimize 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[2]:


X,y=make_blobs(n_samples=500,centers=5,random_state=3)


# In[3]:


plt.scatter(X[:,0],X[:,1],c=y)
plt.grid(True)
plt.show()


# In[4]:


from sklearn.cluster import KMeans


# In[5]:


kmeans=KMeans(n_clusters=5)


# In[6]:


kmeans.fit(X)


# In[7]:


centers=kmeans.cluster_centers_


# In[8]:


plt.scatter(X[:,0],X[:,1])
plt.scatter(centers[:,0],centers[:,1],c='orange')
plt.grid(True)
plt.show()


# In[9]:


pred=kmeans.labels_


# In[10]:


plt.scatter(X[:,0],X[:,1],c=pred)
plt.scatter(centers[:,0],centers[:,1],c='black',marker='*')
plt.grid(True)
plt.show()


# # Failure of K-means algorithm
# 
# - K-means algorithm cannot classify the complex shapes like circle structure etc.

# In[11]:


from sklearn.datasets import make_moons


# In[12]:


X,y=make_moons(n_samples=500,noise=0.1)


# In[13]:


plt.scatter(X[:,0],X[:,1],c=y)
plt.grid(True)
plt.show()


# In[14]:


from sklearn.cluster import KMeans


# In[15]:


kmeans=KMeans(n_clusters=2)


# In[16]:


kmeans.fit(X)


# In[17]:


centers=kmeans.cluster_centers_


# In[18]:


pred=kmeans.labels_


# In[19]:


plt.scatter(X[:,0],X[:,1],c=pred)
plt.scatter(centers[:,0],centers[:,1],c='black',marker='*')
plt.grid(True)
plt.show()


# # DBSCAN:
# 
# -used to give better accuracy to complex structure like moons , circle etc

# In[23]:


from sklearn.cluster import DBSCAN


# In[31]:


# eps is minimum distance between points and 
dbs=DBSCAN(eps=0.2,min_samples=10)


# In[32]:


pred=dbs.fit_predict(X)


# In[33]:


plt.scatter(X[:,0],X[:,1],c=pred)
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




