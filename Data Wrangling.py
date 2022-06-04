#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np


# In[2]:


# look at the data
df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)


# In[3]:


# check for null values
df.isnull().sum()/df.count()*100


# In[4]:


# check data types
df.dtypes


# In[5]:


# Apply value_counts() on column LaunchSite
df['LaunchSite'].value_counts()


# In[6]:


# Apply value_counts on Orbit column
df['Orbit'].value_counts()


# In[7]:


# landing_outcomes = values on Outcome column
landing_outcomes = df['Outcome'].value_counts()
landing_outcomes


# In[8]:


for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)


# In[9]:


bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes


# In[10]:


# landing_class = 0 if bad_outcome
# landing_class = 1 otherwise
#landing_class = [x for x in bad_outcomes if df['Outcome'][x] ]

landing_class = []

for key, value in df['Outcome'].items():
    if value in bad_outcomes:
        landing_class.append(0)
    else:
        landing_class.append(1)


# In[11]:


df['Class']=landing_class
df[['Class']].head(8)


# In[12]:


df.head(5)


# In[13]:


# determine success rate of launch
df["Class"].mean()


# In[14]:


# export the data
df.to_csv("dataset_part_2.csv", index=False)


# In[ ]:




