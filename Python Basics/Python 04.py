#!/usr/bin/env python
# coding: utf-8

# # CLASSES :

# In[14]:


# How to impement a classes :
class Person:
    pass     # Pass means empty class

p= Person() # p is an object of class Person


# In[22]:


class Person:
    name="Ritik"
    def say_hi(self):
        print("Hello",self.name)
        
p=Person()
p.say_hi()

Person.say_hi(p) 


# In[26]:


# initiate by our own value
class Person:
    def __init__(self,name): #initiate by its own
        self.name=name
        
    def say_hi(self):
        print("Hello",self.name)
        
p=Person("Abhishekh")
p.say_hi()


# ## DUNDERS :-

# In[28]:


# Act like a operation overloading like in a c++
class Car:
    def __init__(self,model,mileage):
        self.model=model
        self.mileage=mileage
        
    def __add__(self,other):
        return self.mileage+other.mileage
    
    def __eq__(self,other):
        return self.mileage==other.mileage
    
c1=Car("a",2)
c2=Car("b",2)

print(c1+c2)  # By dunders add
print(c1==c2) # By dunder eq


# In[42]:


class ostream:
    def __init__(self):
        pass
    
    def __lshift__(self,other):
        print(other,end='')
        return self
        
cout= ostream()
cout<<"Ritik Arora"<<"Hello"


# In[49]:


# Making a class dog
class dog:
    #tricks=[]   # As list are mutable so both dog object pointing to same address
    def __init__(self,name="Bruno"):
        self.name=name
        self.tricks=[]  # So both dog object pointing to diffrent adress we create it into a init func
        
    def add_tricks(self,trick):
        (self.tricks).append(trick)
        
a=dog()
b=dog("Tuffy")
a.add_tricks("Fetching")
a.add_tricks("Talk")
print(a.tricks)
print(b.tricks)
print(a.name)
print(b.name)


# ## INHERITANCE : 

# In[54]:


class School_members:
    def __init__(self,name,age,id):
        self.name=name
        self.age=age
        self.id=id
        
    def tell(self):
        print(self.name)
        print(self.age)
        print(self.id)
        
class Students(School_members): # inherit the functionality of class School_members
    def __init__(self,name,age,id,marks):
        # School_members.__init__(name,age,id) can be written as :
        super().__init__(name,age,id)
        self.marks=marks
        
    def tell(self):
        super().tell() # can be written as School_members.tell()
        print(self.marks)
        
class Teachers(School_members): # inherit the functionality of class School_members
    def __init__(self,name,age,id,sal):
        # School_members.__init__(name,age,id) can be written as :
        super().__init__(name,age,id)
        self.sal=sal
        
    def tell(self):
        super().tell()
        print(self.sal)
        
s=Students(age=20,id=63,name="Ritik Arora",marks=81)
t=Teachers("APJ Abdul Kalam",56,22,100000)
s.tell()
print("\n")
t.tell()


# ## Method Resolution :

# In[4]:


class A:
    x=10
class B(A):
    pass
class C(A):
    x=5
class D(C):
    pass
class E(B,D):
    pass

e=E()
e.x
# While making an object E it first go to B rather than A because B is a child of A while has not been traverse yet


# In[ ]:




