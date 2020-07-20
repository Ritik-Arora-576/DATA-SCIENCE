#!/usr/bin/env python
# coding: utf-8

# # File Handing :
# - Read (r) : read an existing file and throw an error if file does not exists
# - Write (w) : Overwrite a file from starting . if file does not exists then it will create a new file
# - Append (a) : Write a file from ending . If file does not exists the it will create a new file
# - Read and Write (r+) : First read and the overwrite it . Error if not exsts
# - Write and Read (w+) : first write and the read . Created if not exists
# - Append and read (a+) : first append then read. Created if not exists

# ### Steps :
# - First open a file in any mode(read, write ,append.....)
# - if , file = open("hello.txt","r") then file open in read mode and, file.read()
# - close a file by file.close()

# In[2]:


# read a file
file = open("hello.txt","r")
file.read()


# In[3]:


file.close()


# In[9]:


# write and read a file
file = open("hello.txt","w+")
file.write("I am a Ritik Arora") # cursor move to last point so file can read nothing
file.seek(0) # move cusror to 0th position
print(file.read(8)) # read upto 8 bit
print(file.read()) # cursor go to 8th position and read after it
file.close()


# In[21]:


with open("hello.txt","r") as file:
    print(file.read())
    file.seek(0)
    print("\n",file.readline()) # read only a lines
    print(file.readlines()) # read multiple lines
# file closes automatically


# ## ERROR HANDLING :

# In[24]:


def div(a,b):
    try:    # if no error the run this . if yes then move to except block
        return a/b
    except:
        print("Error") # this except is use to handling the case of error
        
print(div(10,2))
div(10,0)


# In[25]:


print(10/0)


# In[29]:


try:
    a=int("Ritik")
    print(10/0)
except ZeroDivisionError:
    print("You try to divide it from 0")
except ValueError:
    print("Error")


# In[27]:


a=int("a")


# In[34]:


try:
    print(10/0)
except Exception as e: # tells us about which exception is error
    print(e)
    print(type(Exception)) # Exception is a class while e is an object of tha class


# In[38]:


try:
    raise Exception("Hello")
except Exception as e:
    print(e)


# In[43]:


class MyException(Exception):
    def __init__(self,message):
        self.message=message
    def __str__(self):
        return self.message

try:
    raise MyException("There is some error") # error raised by us
except Exception as e:
    print(e.message)


# - try : throw eigther error or not
# - else : if no error occur in try block then else will be executed
# - except : if error occur in try block then except will be executed
# - finally : if try block error or not , return come first or not finally will be excute
#             in every condition

# In[46]:


try:
    print(1/0)
except:
    print("There is must be some error")
else:
    print("Try case is passes")
finally:
    print("By by")


# In[53]:


with open("ritik.txt","r") as file:
    try:
        print(file.read())
    except :
        print("File does not exists")
    else:
        print("File is existed")
    finally:
        print("Ok bye")

