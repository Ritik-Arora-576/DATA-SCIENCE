#!/usr/bin/env python
# coding: utf-8

# In[5]:


#writing hello world programme
print("Hello World !")


# In[8]:


''' this is a 
multiline 
comment'''
print("Hello again")


# In[14]:


a=5
b=5.7
c=1+9j
print(a," ",b," ",c)
#for changing a line
print("\n")
#tell us about type of variable
type(a)
type(b)
type(c)


# In[15]:


#For single line strings
a="I learn python"
print(a)
a='I also learn python'
print(a)


# In[18]:


#For multiple line strings
a="""Python is very 
easy to
understand"""
print(a,"\n")

a='''python is 
a future for
developers'''

print(a)
type(a)


# # Arithmatic Operators

# In[3]:


a=10
b=21
print(a+b)
print(a-b)
print(a*b)
# In python we get float value
print(a/b)
print(a%b)
#Division integer by integer we use //
print(a//b)
#Power or exponential function
print(a**b)
#Division upto 4 place of decimal
print("%0.4f"%(a/b))


# # MULTIPLE ASSIGNED VALUES

# In[22]:


a,b,c=1,2,"Hello"
print(a,b,c)


# ## CONDITIONAL STATEMENTS

# In[4]:


a=-10

if a>0:
    print("Positive")
elif a<0:
    print("Negetive")
else:
    print("Zero")


# ## NESTED IF-ELSE

# In[11]:


a=10

#Spacing tells us about code of particular statement

if a>0:
    print("Number is positive")
    if a%5==0:
        print("Number is divisible by 5")
else:
    print("Number is negetive")


# ## LOOPING

# In[23]:


# while loop
a=10

while a>0:
    print(a)
    a-=1

print("\n")
# for loop
for i in "python":
    print(i)
    
print("\n")

for i in range(5):
    print(i)
    
print("\n")

for i in range(2,10):
    print(i)
    
print("\n")

for i in range(1,10,2):
    print(i)


# In[14]:


# how to know about ASCII value of a character
ord("A")


# ord("0")   #ASCII value of integer

# ## LOGICAL OPERATOR
# - and (&&)
# - or (||)
# - not (!)

# In[15]:


isinstance(True,int) # True and False are the integer value


# In[16]:


isinstance(True,str) #True and false are not the string value


# In[18]:


'''
value of True =1
value of False=0
'''
5+True


# In[19]:


5*False


# In[22]:


True and False


# In[23]:


True or False


# In[24]:


not True


# In[25]:


not False


# In[27]:


5 or 7 # 1 or B = 1 (where first element is 5 thats why output is 5)


# In[29]:


5 and 7 # 1 and B = B (second element is 7)


# In[30]:


not 5


# In[ ]:




