#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
# creating an array
a=np.array([1,2,3,4,5,6])
print(a)
print(np.shape(a)) # Dimensions of array


# In[6]:


# Creating a 2D array
b=np.array([[1,2,3,4,5],[1,2,3,4,5]])
print(b)
print(np.shape(b))
print(b[1][3])


# In[11]:


# Making a zero array(Array containing all the elements zero)
a_zero=np.zeros((3,3))
print(a_zero)
# Making one array
a_one=np.ones((5,5))
print(a_one)
# Making an array of same constant
constant=int(input())
a_constant=np.full((3,3),constant)
print(a_constant)


# In[29]:


#Making identity Matrix :
a=np.eye(4) # Identity matrix of 4x4
print(a)
# Making Random Matrix
random_matrix = np.random.random((2,3))
print(random_matrix)


# In[18]:


print(random_matrix[:,1]) # Print all the elements of 1st column
# Updation
random_matrix[1,1:]=1
print(random_matrix)


# In[28]:


#data Types:
print(random_matrix.dtype)
# Used to convert data Type into integer value
arr=np.ones((2,4),dtype=np.int64)
print(arr)


# ## Mathematical Operations :
# - add
# - subtract
# - multiply
# - divide
# - dot
# - sum

# In[32]:


a=np.array([1,2,3,4])
b=np.array([5,6,7,8])
print(a+b)
print(np.add(a,b))
print(a-b)
print(np.subtract(a,b))


# In[37]:


x=np.array([[1,2],[3,4]])
y=np.array([[1,2],[3,4]])
print(x*y)
print(np.multiply(x,y)) # Multiply corresponding elements of 2 matrix
print(x/y)
print(np.divide(x,y))# Divide corresponding elements of 2 matrix


# In[40]:


# dot product of 2 matrices: (i.e Matrix Multiplication)
print(np.dot(a,b))
print(np.dot(x,y))


# In[43]:


print(np.sum(x)) # sum all the elements inside a matrix
print(np.sum(x,axis=0)) # sum of all the elements of matrix along the column
print(np.sum(x,axis=1)) #sum of all the elements of a matrix along row


# In[47]:


# Stacking of two arrays:
a=np.array([1,2,3,4])
b=np.array([1,2,3,4])
print(np.stack((a,b),axis=1))
print(np.stack((a,b),axis=0))
a=np.stack((a,b),axis=0)


# In[49]:


# Reshaping of an array:
print(a.reshape((8,)))


# In[50]:


print(a.reshape(4,-1))


# In[51]:


print(a.reshape(-1,4))


# ## STATISTICAL FUNCTION :
# - min
# - max
# - mean
# - median
# - average
# - variance
# - standard deviation

# In[55]:


x=np.array([[1,2,3,4],[5,6,7,8]])
print(np.max(x))
print(np.min(x))
print(np.min(x,axis=0))
print(np.min(x,axis=1))


# In[57]:


print(np.mean(x))
print(np.mean(x,axis=0))


# In[59]:


x=np.array([1,2,3,4,5,6,7])
print(np.median(x))


# In[66]:


# In average we have to assign the wieghts of each elements in array
wieghts=np.array([1,2,3,4,5,6,7])
print(np.average(x,weights=wieghts))


# In[67]:


print(np.std(x))


# In[68]:


print(np.std(x)**2)
print(np.var(x))


# In[69]:


print(np.sqrt(x))


# ### Random Module :

# In[84]:


x=np.arange(10) + 5
print(x)


# In[85]:


y=np.random.shuffle(x)
print(y)


# In[90]:


print(np.random.rand(2,3))


# In[91]:


print(np.random.randn(2,3))


# In[92]:


print(np.random.randint(5,10,4)) # print 4 random numbers in the range between 5 to 10


# In[94]:


print(np.random.choice([1,2,3,4,5,6]))


# In[108]:


x=np.array([1,1,1])
y=np.array([1,1,1])
print(np.hstack((x,y)))


# In[ ]:




