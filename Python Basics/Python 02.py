#!/usr/bin/env python
# coding: utf-8

# # FUNCTIONS IN PYTHON :

# - FUNCTION WITH NO ARGUEMENTS

# In[2]:


# knocking of door of sheldon at penny door

def sheldon_knocks():
    print("Knock Knock Knock Penny")
    print("Knock Knock Knock Penny")
    print("Knock Knock Knock Penny")
    
sheldon_knocks() #declaration of a function


# - FUNCTIONS WITH ARGUEMENTS

# In[10]:


def sheldon_knocks(name,no_of_knocks=3): 
    for i in range(no_of_knocks):
         print("knock knock knock ",name)
    
sheldon_knocks("Penny",5) 


# In[12]:


sheldon_knocks("Penny") # Without defining number of knocks it takes the default value inside a function


# ### Return value of a Function :

# In[14]:


def add(a,b):
    return a+b

x=add(1,9)
print(x)


# - try , except , and finally

# In[15]:


# finally block always executes even after return statement comes first

def div(a,b):
    try:  # If a/b is defined
        return a/b
    except: # If a/b is not defned like in 10/0
        print("Error")
    finally: # Executes surely even after return statement
        print("Wrapping Up")
        
div(10,2)


# In[16]:


div(10,0)


# - Local and Global Variables

# In[33]:


x=10 # This is a global variable

def show():
    y=7 # This is a local variable
    print(x) 
    print(x)
    print(y)
    
show()
print(y) # This shows error because scope of y ended in a show function


# In[37]:


# If we want to change the value of global variable x
x=10

def show():
    global x # Used to define interpreter that we use global variable x
    x+=5
    print(x)
    
show()


# In[38]:


del x


# In[44]:


def outer():
    x=10
    
    def inner():
        nonlocal x  # x is not a supreme global variable like in upper case thats why we use nonlocal so it will see x out of local scope
        x+=5
        print(x)
        
    inner()
    print(x)
    
outer()


# ## ARGUEMENTS

# In[45]:


def show(a,b,c):
    print(a)
    print(b)
    print(c)
    
show("Hello","Ritik","Arora")


# In[46]:


def show(a,b,c):
    print(a)
    print(b)
    print(c)
    
show(b="Hello",a="Ritik",c="Arora") # This is a keyworded arguement i.e hello stores in b etc.


# In[48]:


# Packing Arguement
def show(*arg):
    print(arg)
    
show("Hello","Ritik","Arora","in","Python") # These all are packed in arg parameter


# In[5]:


def show(a,b,c,*arg,d="Ritik",e="Arora",**arg2):
    print(a,b,c)
    print(arg)
    print(d,e)
    print(arg2) #Use for undefined variable arguement
    
show(1,2,3,"My","First","Programme",e="Roshan",name="Ayushmaan")


# ## LAMBDA FUNCTIONS

# In[58]:


# Lambda function is a one line function

'''def add(a,b):
    return a+b

add(1,2)'''

# It can be written in one line :
add = lambda a , b : a+b
add(1,2)


# In[ ]:


a=[("Ram",50),("Shyam",100),("Ghanshyam",10),("Balram",30)]
 #If we want to do sorted according to 1st elememnt
sorted(a)

#If we want to do sorted according to 2nd element
def key(x):
    return x[1]

sorted(a,key = lambda x:x[1])
sorted(a,key=key)


# # DECORATORS

# In[2]:


# Making a dictionary

users={
    "Ritik" : "Ritik123",
    "Jatin" : "Mentor",
    "prateek" : "Mentor"
}
def show(username,password):
    if username in users and users[username]==password:
        print("You are a part of data science")
        
    else:
        print("Not authenticated")
        
show("Ritik","Ritik123")


# In[10]:


# above method is little lengthy

users={
    "Ritik" : "Ritik123",
    "Arora" : "Coding Blocks"
}

def login(func): # Here login function use functional object i.e function as an object
    def wrapper(username,password,*arg,**kwarg):
        if username in users and users[username]==password:
            # User is authenticalted 
            func(*arg,**kwarg)
            
        else:
            print("Oops ! You are not authenticated")
            
    return wrapper


# In[12]:


# if we want to run add function by using username and password
def add(a,b):
    print(a+b)
    
add=login(add)

add("Ritik","Ritik123",1,2)


# In[15]:


# Decorators
@login
def add(a,b):
    print(a+b)
    
add("Arora","Coding Blocks",1,2)


# In[ ]:




