#!/usr/bin/env python
# coding: utf-8

# In[67]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[68]:


final_img_arr=np.zeros((430,430,3))


# In[69]:


plt.imshow(final_img_arr)
plt.axis('off')
plt.show()


# In[70]:


top_left=cv2.imread('top_left.jpg')
top_left=cv2.cvtColor(top_left,cv2.COLOR_BGR2RGB)
top_right=cv2.imread('top_right.jpg')
top_right=cv2.cvtColor(top_right,cv2.COLOR_BGR2RGB)
bottom_left=cv2.imread('bottom_left.jpg')
bottom_left=cv2.cvtColor(bottom_left,cv2.COLOR_BGR2RGB)
bottom_right=cv2.imread('bottom_right.jpg')
bottom_right=cv2.cvtColor(bottom_right,cv2.COLOR_BGR2RGB)
center=cv2.imread('center.jpeg')
center=cv2.cvtColor(center,cv2.COLOR_BGR2RGB)


# In[71]:


top_left=cv2.resize(top_left,(200,200))
top_right=cv2.resize(top_right,(200,200))
bottom_left=cv2.resize(bottom_left,(200,200))
bottom_right=cv2.resize(bottom_right,(200,200))
center=cv2.resize(center,(200,200))


# In[72]:


plt.imshow(top_left)
plt.show()
plt.imshow(top_right)
plt.show()
plt.imshow(bottom_left)
plt.show()
plt.imshow(bottom_right)
plt.show()
plt.imshow(center)
plt.show()


# In[73]:


top_left=np.array(top_left,dtype=np.float16)/255.0


# In[74]:


top_right=np.array(top_right,dtype=np.float16)/255.0


# In[75]:


bottom_left=np.array(bottom_left,dtype=np.float16)/255.0


# In[76]:


bottom_right=np.array(bottom_right,dtype=np.float16)/255.0


# In[77]:


center=cv2.resize(center,(100,100))
center.shape


# In[78]:


center=np.array(center,dtype=np.float16)/255.0


# In[79]:


x1=np.zeros((100,10,3))
x2=np.zeros((10,120,3))


# In[80]:


final_img_arr[10:210,10:210]=top_left
final_img_arr[10:210,220:420]=top_right
final_img_arr[220:420,10:210]=bottom_left
final_img_arr[220:420,220:420]=bottom_right
final_img_arr[165:265,165:265]=center
final_img_arr[165:265,155:165]=x1
final_img_arr[165:265,265:275]=x1
final_img_arr[155:165,155:275]=x2
final_img_arr[265:275,155:275]=x2


# In[81]:


plt.figure(figsize=(7,7))
plt.imshow(final_img_arr)
plt.axis('off')
plt.show()


# In[82]:


final_img_arr.shape


# In[84]:


final_img_arr=final_img_arr.reshape((-1,3))


# In[85]:


final_img_arr.shape


# In[86]:


import pandas as pd


# In[87]:


df=pd.DataFrame(final_img_arr,columns=['r','g','b'])


# In[88]:


df.to_csv('Instagram_collage.csv',index=False)


# In[95]:


plt.imshow(final_img_arr.reshape((430,430,3)))


# In[ ]:




