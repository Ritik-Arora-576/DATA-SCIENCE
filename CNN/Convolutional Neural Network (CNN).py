#!/usr/bin/env python
# coding: utf-8

# ## Preparing the image :

# In[1]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


plt.style.use('seaborn')
img=cv2.imread('bottom_left.jpg')
img_=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img_=cv2.resize(img_,(100,100))
img_gray=cv2.resize(img_gray,(100,100))
plt.axis('off')
plt.imshow(img_)
plt.title('Image '+str(img_.shape))
plt.show()
plt.imshow(img_gray,cmap='gray')
plt.title('Image '+str(img_gray.shape))
plt.axis('off')
plt.show()


# ## Apply filters on image :

# In[3]:


def convolution(image_arr,filter_arr):
    hieght=image_arr.shape[0]
    width=image_arr.shape[1]
    
    filter_hieght=filter_arr.shape[0]
    filter_width=filter_arr.shape[1]
    
    new_img=np.zeros((hieght-filter_hieght+1,width-filter_width+1))
    
    for row in range(hieght-filter_hieght+1):
        for col in range(width-filter_width+1):
            for i in range(filter_hieght):
                for j in range(filter_width):
                    new_img[row,col]+=image_arr[row+i,col+j]*filter_arr[i,j]
                    
                if new_img[row,col]>255:
                    new_img[row,col]=255
                    
                elif new_img[row,col]<0:
                    new_img[row,col]=0
                    
    return new_img


# In[4]:


blur_filter=np.ones((3,3))/9.0   # filter use for blurring the image
output_img=convolution(img_gray,blur_filter)


# In[5]:


plt.style.use('seaborn')
plt.axis('off')
plt.imshow(output_img,cmap='gray')
plt.title('Image '+str(output_img.shape))
plt.show()


# In[6]:


edge_filter=np.array([[1,0,-1],
                     [1,0,-1],
                     [1,0,-1]]) # shows only edges of an image
output_img=convolution(img_gray,edge_filter)

plt.style.use('seaborn')
plt.axis('off')
plt.imshow(output_img,cmap='gray')
plt.title('Image '+str(output_img.shape))
plt.show()


# ### Padding :
# 
# - We add rows and col around an image
# 
# - after padding dimension of an output image is :
#     * h_new = h_old - f + 2*pad +1
#     * w_old = w_old -f + 2*pad +1
#     
# ### Stride :
# 
# - We skip filters by certain number of channels :
# 
# - after striding :
# 
#     * h_new = (h_old - f + 2*pad)/stride +1
#     * w_old = (w_old -f + 2*pad)/stride +1

# In[7]:


# implementing padding 

new_img=np.pad(img_,((10,10),(20,20),(0,0)),mode='constant',constant_values=0) 
# image array , padding size along hieght width and channels , filling constant values at that channel


# In[8]:


plt.imshow(img_)
plt.title('Original Image '+str(img_.shape))
plt.axis('off')
plt.show()

plt.imshow(new_img)
plt.title('Image after padding '+str(new_img.shape))
plt.axis('off')
plt.show()


# ### Pooling :
# 

# In[9]:


def pooling(arr,stride=2,f=2,mode='max'):
    h=arr.shape[0]
    w=arr.shape[1]
    
    a1=int(((h-f)/stride)+1)
    a2=int(((w-f)/stride)+1)
    new_arr=np.zeros((a1,a2))
    
    for row in range(a1):
        for col in range(a2):
            
            X=arr[row*stride:row*stride + f,col*stride:col*stride + f]
            
            if mode=='max':
                new_arr[row,col]=np.max(X)
                
            else:
                new_arr[row,col]=np.mean(X)
                
    return new_arr


# In[ ]:




