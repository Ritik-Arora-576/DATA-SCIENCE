#!/usr/bin/env python
# coding: utf-8

# # DATA ACQUISITION (WEB SCRAPPING) :

# In[1]:


from urllib.request import urlopen
# Copying the URL of web from which u want to scrap data
url="https://en.wikipedia.org/wiki/Android_version_history" 
data=urlopen(url) # Aquiring the data by opening the URL
url_html=data.read() # Read URL in the form of HTML
print(url_html)
data.close #closing the data


# ## BEAUTIFUL SOUP:

# In[2]:


# Beautiful soup are use for extracting specific data from the set of HTML
from bs4 import BeautifulSoup as soup
# BeautifulSoup is a class
android_soup=soup(url_html,'html.parser')
print(android_soup)


# In[3]:


# Finding the main heading of a web page
main_heading=android_soup.find_all('h1',{}) # h1 represent heading in HTML and {} use for filteration
print(main_heading)


# In[5]:


for i in main_heading:
    print(i.text)  # i.text used to convert HTML codes into text form


# In[6]:


# Extracting all tables 
all_tables=android_soup.find_all('table',{'class':'wikitable'})
print(all_tables)


# In[9]:


for i in all_tables:
    print(i.text)


# In[15]:


#print the heading of a table 0
table_headings=all_tables[0].find_all('th',{})
for i in table_headings:
    print(i.text)

# Store in the form of list
table_headings=[i.text[:-1] for i in table_headings]
print(table_headings)


# In[18]:


#print all the rows of the table 0
all_rows=all_tables[0].find_all('tr',{})
for i in all_rows:
    print(i.text)


# In[85]:


# store all the rows in the form of list
table_rows=[]
for row in all_rows:
    current_row=[]
    x=row.find_all('td',{})
    for idx,data in enumerate(x):
            if idx==2:
                current_row.append(data.text[:-1].replace(',',''))
            else:
                current_row.append(data.text[:-1]) # -1 is used to remove last element '\n'
    table_rows.append(current_row)
print(table_rows)


# ### CSV FILE : 

# In[86]:


file_name="android_version.csv"
with open(file_name,'w',encoding='utf-8') as f:
    heading_string=', '.join(table_headings) # join the list
    heading_string+='\n'
    f.write(heading_string)
    for row in table_rows:
        row_string=""
        row_string+=', '.join(row)
        row_string+='\n'
        f.write(row_string)


# ## Pandas :

# In[87]:


# Pandas are use to convert CSV files into tabular informations
import pandas as pd
panda_data=pd.read_csv("android_version.csv")
panda_data.head(10) #print first 10 rows


# In[88]:


panda_data.iloc[5][0]

