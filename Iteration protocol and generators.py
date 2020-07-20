#!/usr/bin/env python
# coding: utf-8

# ## Iterations :

# In[3]:


x=[1,2,3,4]
x_iter=iter(x)
next(x_iter)


# In[21]:


# Here iterable and iterator encapsulate in one class
# so Yrange is both iterable and iterator
class Yrange:
    def __init__(self,n):
        self.n=n
        self.i=0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i<self.n:
            i=self.i
            self.i+=1
            return i
        else:
            raise StopIteration()
y=Yrange(10)
print(list(y))
print(list(y))  


# In[19]:


# Iterator and iterable both are in different classes
class Yrange(y_range):
    def __init__(self,n):
        self.n=n
    def __iter__(self):
        return y_range(self.n)
    
class y_range:
    def __init__(self,n):
        self.i=0
        self.n=n
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i<self.n:
            i=self.i
            self.i+=1
            return i
        else:
            raise StopIteration()
            
y=Yrange(10)
print(list(y))
print(list(y))


# ## Fibonacci Series :

# In[23]:


class Fib:
    def __init__(self):
        self.prev=0
        self.curr=1
    def __iter__(self):
        return self
    def __next__(self):
        value=self.curr
        self.curr+=self.prev
        self.prev=value
        return self.curr
    


# In[25]:


f=iter(Fib())


# In[35]:


next(f)


# ### GENERATORS :

# In[36]:


def fib():
    prev , curr = 0 , 1
    while True:
        yield curr
        prev , curr = curr , prev+curr
f=fib()


# In[45]:


next(f)


# ### GENERATOR EXPRESSION:

# In[46]:


gen = (x**2 for x in range(10))


# In[53]:


next(gen)


# In[54]:


a=[1,2,3,4,5]
x_iter=iter(a) 


# In[64]:


print(x_iter)


# In[ ]:




