#!/usr/bin/env python
# coding: utf-8

# # IMAGE PRESENTATION :

# In[97]:


# Every pixel contain 3 cells in the form of tupples name (RGB)
import numpy as np
T=np.zeros((5,10,3),dtype=np.uint8) # we choose unsigned integer because we require only 8 bits to store data
T[:,:,2]=255
#print(T)
import matplotlib.pyplot as plt
print(plt.imshow(T))
T[:,:,1]=255
print(plt.imshow(T))


# # Transpose :

# In[84]:


x=np.zeros((4,2))
print(x)
print(np.transpose(x)) # Transpose of a matrix


# In[89]:


print(np.shape(T))
T1=np.transpose(T,axes=(1,2,0))
print(np.shape(T1))


# # Broadcasting:
# - Adding vectors with scalar
# - Adding vectors with vectors

# In[98]:


x=np.array([1,2,3,4,5,6])
print(x+4)


# In[99]:


X=np.array([[2,3,4,5,6,7],[5,6,7,8,9,10]])
print(x+X)


# ## Norms :

# In[6]:


import numpy as np
x=np.array([3,-4])
print(np.linalg.norm(x)) # print the norm of order 2
print(np.linalg.norm(x,ord=1)) # print the sum of all absolute values in vector
print(np.linalg.norm(x,ord=np.inf)) # print maximum element


# ## DETERMINANT:

# In[8]:


a=np.array([[1,2],[3,4]])
print(np.linalg.det(a))


# In[10]:


'''
INVERSE OF
A MATRIX
'''
ainv=np.linalg.inv(a)
print(ainv)
print(np.dot(a,ainv))


# In[11]:


# solve linear of equations
a=np.array([[1,2],[3,4]])
b=np.array([5,7])
print(np.linalg.solve(a,b))


# In[ ]:





# In[ ]:




