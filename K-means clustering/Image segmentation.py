#!/usr/bin/env python
# coding: utf-8

# In[65]:


import matplotlib.pyplot as plt
import cv2
import numpy as np


# In[66]:


im=cv2.imread('elephant.jpg')
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)


# In[67]:


plt.imshow(im)
plt.show()


# In[68]:


all_pixels=im.reshape((-1,3))


# In[69]:


from sklearn.cluster import KMeans
max_clusters=7
kmeans=KMeans(n_clusters=max_clusters)


# In[70]:


kmeans.fit(all_pixels)


# In[71]:


centers=kmeans.cluster_centers_


# In[72]:


centers


# In[73]:


# convert the centers of the clusters into integer values
centers=np.array(centers,dtype='uint8') # we used unsigned int because pixel values are always positive


# In[74]:


centers


# ## Extracting the most dominant color :

# In[75]:


i=1

plt.figure(figsize=(8,2))

colors=[]

for each_col in centers:
    colors.append(each_col)
    plt.subplot(1,7,i)
    i+=1
    
    a=np.zeros((100,100,3),dtype='uint8')
    
    a[:,:,:]=each_col
    plt.axis('off')
    plt.imshow(a)
    
plt.show()


# In[76]:


colors


# In[77]:


kmeans.labels_


# ## Draw a segmented image :

# In[78]:


segmented_image=np.zeros(all_pixels.shape,dtype='uint8')


# In[79]:


for i in range(segmented_image.shape[0]):
    segmented_image[i]=colors[kmeans.labels_[i]]


# In[80]:


segmented_image=segmented_image.reshape(im.shape)


# In[81]:


plt.imshow(segmented_image)
plt.axis('off')
plt.show()


# In[ ]:




