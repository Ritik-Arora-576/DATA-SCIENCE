#!/usr/bin/env python
# coding: utf-8

# # DATA STRUCTURES :

# ### STRINGS :

# In[3]:


# Collection of characters

a="Ritik"
b='Ritik Arora'
# For multilines strings
c="""
    Hi this 
    is Ritik 
    Arora
    """
# For multilines strings
d='''           
   Hi Again !
   I am learning 
   strings
   '''
print(a)
print(b)
print(c)
print(d)

ord("A") # Teels us about ASCII value
chr(65) # Teels us about char related to ASCII value 65 i.e A


# - Operation in strings (Formatting):

# In[7]:


a=1
b=2
c=3
print(str(a) + '-' + str(b) + '-' + str(c)) # Where str(variable) convert variable of any datatype into str
print("%d-%d-%d" % (a,b,c))
#but if we replace a=1 by "Ritik"
a="Ritik"
print("%d-%d-%d" % (a,b,c)) # Error would be come because %d is a format specifier for int


# In[11]:


# Flexible Formatting
print("{}-{}-{}".format(a,b,c))
# We can assign positions also
print("{1}-{2}-{0}".format(a,b,c))

print("{firstname},{Lastname}".format(firstname="Ritik",Lastname="Arora"))

# or according to latest versions
firstname="Ritik"
lastname="Arora"
print(f"{firstname} , {lastname}")


# - strip function : (eleminates all blanck spaces in string)

# In[14]:


a="         Ritik Arora         "
print(a)
print(a.strip())


# In[15]:


a=input()


# In[17]:


a.strip()=="yes"


# - split function :

# In[29]:


a=input()


# In[30]:


print(a)
print(a.split())


# - replace function :

# In[25]:


a="aaabbbcccaaddd"
a.replace("a","z") # replaces all the a by z


# - count function :

# In[35]:


a="azzbbizrtz"
a.count('z')


# In[36]:


a.count('zz')


# ### Lists :

# In[38]:


a=[1,2,3,4,5,6,7]
print(a[2])


# In[46]:


# lists can be hetrogeneous i.e made of int , str , dict , tupples etc. in a single list
a=[1,"Ritik",3.7,print,(1,2,3),{"Name" : "Ritik" , "College" : "DTU"}]
print(a)
a[3]("Hello World")
print("\n")
for i in a[5]:
    print(i,a[5][i])


# In[54]:


len(a) # length of list
for i in a:
    print(i)

a+a


# In[52]:


a*3


# In[59]:


a=[0,1,2,3,4]
print(a[0],a[-1])
print(a[-2])
# sublist
print(a[1:4])
print(a[0:5:2]) # increament my 2
print(a[:]) # By default starts with starting point and ends at ending point


# In[66]:


# check is a given string is palindrome or not
a="jatin"
print(len(a))
print(a[2])
if a[::-1]==a:
    print("True")
else:
    print("False")


# - Update inside a list :

# In[69]:


a=[1,2,3,4,5,6]
a.insert(1,"Ritik")
print(a)
a.append("Arora") #insert the object into last index of the list
print(a)


# - Delelte an element from the list :

# In[76]:


a=[1,2,3,4,5,6]
a.pop(2) # If we want to delete a particular index from the list
print(a)
a.remove(2) # If we want to remove an object 
print(a)

del a[0] # Delete the object from the memory
print(a)
del a # Delete a from the memory
print(a)


# - sort a list :

# In[80]:


a=[7,3,5,9,11,10]
a.sort()
print(a)
b=[19,56,21,32,11]
sorted(b)


# - Reverse a list :

# In[85]:


a.reverse()
print(a)


# ### TUPPLES :

# In[86]:


def show(*arg):
    print(arg)
show(1,2,3,4,5) # this return a tupple


# In[93]:


# Assigning a value
a = (1,2,3,4,5,6)
print(a)
#or
b= 1,2,34,5,6
print(b)
print(a[0])
print("\n")
a,b,c = 1,2,3
print(a)
print(b)
print(c)


# In[96]:


# swapping two integers
a=3
b=5
print("Before swapping : ",a,b)
a,b = b,a #Assigning the values
print("After swapping : ",a,b)


# In[101]:


a=(1,2,3,4,5)
a=list(a) #Convert a tuple into list
print(a)
b=[1,2,3,4,5]
b=tuple(b) #Conert a list into tupple
print(b)


# In[103]:


def add_subtract(a,b):
    return (a+b , a-b) # returning a tupple
sum , diff = add_subtract(6,9)
print("Sum : ",sum)
print("Difference : ",diff)


# ### DICTIONARIES :

# In[4]:


# Dictionaries are made up of key values pairs
a = {"Name" : "Ritik Arora" , "Roll Number" : 63 , "Skills" : ("C++","Python","ML","C")}
print(a["Name"],"\n")

for key in a:
    print(key," : ",a[key])


# - Operations in dictionaries :
#    1. keys()
#    2. values()
#    3. items()
#    4. get()
#    5. clear()

# In[ ]:


# 1. keys()
for i in a.keys():
    print(i)  # Print all keys 
    
# 2. values()
for i in a.values():
    print(i) # Print all values
    
# 3. items()
for i in a.items():
    print(i) # Print tuple of key and value pairs
    
# 4. get()
a.get("Name") # Get the value of key entered

# 5. clear()
a.clear() # Clears the dictionary


# ### SETS :

# In[ ]:


a={1,2,3,4,5,6} # this is a set
print(a)


# In[ ]:


# to create empty sets
a=set()
print(a)


# In[ ]:


a={9,9,4,10,4,32,100,54,32}
print(a)


# In[ ]:


a={1,2,3,4}
b={3,4,5,6}
a.intersection(b)
a.union(b)
a-b


# In[ ]:


print(a[2]) # not valid


# In[ ]:


for i in a:
    print(i) # sets are iterable


# In[ ]:




