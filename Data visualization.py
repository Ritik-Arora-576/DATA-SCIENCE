#!/usr/bin/env python
# coding: utf-8

# # DATA VISUALIZATION :

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
a=np.zeros((16,20,3))
a[:,:,1]=255
plt.imshow(a)


# In[3]:


plt.style.use('Solarize_Light2')
x=np.arange(10)
y1=x**2
y2=3*x+5
plt.xlabel("Time")
plt.ylabel('COVID-19 Cases')
plt.title("Corona cases over time")
plt.plot(x,y1,color='red',label='Corona cases',marker='o')
plt.plot(x,y2,color='blue',label='Deaths',linestyle='dashed',marker='*')
plt.legend()
plt.show()


# In[4]:


plt.style.available


# In[5]:


price=np.array([1,2,3,4])**2
plt.plot(price)
plt.show()


# # SCATTER :

# In[6]:


plt.figure(figsize=(2,2)) # Modify the size of the graph
x=np.arange(10)
y1=x**2
y2=2*x+4
plt.scatter(x,y1,color='red',marker='^',label='Profit')
plt.scatter(x,y2,color='green',marker='o',label='customers')
plt.xlabel('Time')
plt.ylabel('Supply')
plt.title('Supple Rate')
plt.legend()
plt.show()


# ## Bar Graphs :

# In[3]:


plt.style.use("dark_background")
plt.figure(figsize=(7,7))
year=np.array([1,2,3,4])
price=np.array([70,40,60,90])
taxes=np.array([11,9,17,15])
plt.bar(year,price,width=0.25,color='orange',label='Prices',tick_label=['Gold','Iron','Silver','Platinum'])
plt.bar(year,taxes,width=0.25,align='edge',color='green',label='Taxes')
plt.xlabel("Years")
plt.ylabel("Prices in 1000 USD")
plt.ylim(0,120)
plt.xlim(0,6)
plt.legend()
plt.show()


# ## PIE CHARTS :

# In[18]:


plt.style.use('seaborn')
plt.figure(figsize=(6,6))
subjects='Maths','Physics','Chemistry','English','Computer Science'
wieghtage=[100,70,70,50,80]
plt.pie(wieghtage,labels=subjects,colors=['red','green','blue','yellow','orange'],shadow=True,explode=(0.1,0,0,0,0),autopct='%1.2f%%')
plt.title('Percentage of subjects in class 12th board')
plt.show()


# # HISTOGRAM :

# In[41]:


x_std=np.random.randn(100) # generates random number between -1 and 1
sigma=8
mean=70
x_data=(x_std*sigma)+mean # x_std = (x-mean)/sigma
x_data1=(x_std*5)+30
print(x_data)


# In[47]:


plt.style.use('seaborn')
plt.hist(x_data,color='red',alpha=0.8,label='English')
plt.hist(x_data1,color='blue',alpha=0.8,label='Chemistry')
plt.title('histogram')
plt.xlabel('Marks')
plt.ylabel('Frequency of students')
plt.title('Distribution of Marks in class')
plt.legend()
plt.show()


# In[43]:


get_ipython().run_line_magic('pinfo', 'plt.hist')


# In[ ]:




