#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("data.csv",encoding = 'ISO-8859-1')
df.head()


# In[3]:


df.info()


# ## Statistical Summary

# In[4]:


df.describe()


# In[5]:


df.describe(include=object)


# ## MISSING VALUE TREATMENT

# In[6]:


df.isnull().sum()


# In[7]:


df = df.loc[df['Quantity'] > 0]
df = df.loc[df['UnitPrice'] > 0]


# In[8]:


df.isnull().sum()


# In[9]:


df.loc[df['CustomerID'].isna()].head()


# In[10]:


df.nunique()


# In[11]:


df.shape


# In[12]:


df = df.dropna(subset=['CustomerID'])


# In[13]:


df.shape


# In[14]:


df.isnull().sum()


# ## COLLABORATIVE FILTERING
# 
# The models created by collaborative filtering techniques are based on the prior actions of a user (things previously chosen or purchased, and/or numerical ratings given to those items), as well as comparable choices made by other users. Then, this model is used to forecast the ratings for things or items themselves that the user could be interested in.

# In[15]:


customer_item_matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity',aggfunc='sum')
customer_item_matrix.head()


# In[16]:


customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
customer_item_matrix.head()


# In[17]:


customer_item_matrix.shape


# ## (A) Creating User-to-User Similarity Matrix

# In[18]:


from sklearn.metrics.pairwise import cosine_similarity

user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
user_user_sim_matrix


# In[19]:


user_user_sim_matrix.shape


# In[20]:


user_user_sim_matrix.columns = customer_item_matrix.index

user_user_sim_matrix['CustomerID'] = customer_item_matrix.index

user_user_sim_matrix = user_user_sim_matrix.set_index('CustomerID')
user_user_sim_matrix.head()


# In[21]:


user_user_sim_matrix.loc[12557].sort_values(ascending=False)


# In[22]:


items_bought_by_12557 = set(customer_item_matrix.loc[12557].iloc[customer_item_matrix.loc[12557].to_numpy().nonzero()].index)
items_bought_by_12557


# In[23]:


items_bought_by_12431 = set(customer_item_matrix.loc[12431.0].iloc[customer_item_matrix.loc[12431.0].to_numpy().nonzero()].index)
items_bought_by_12431


# In[24]:


items_to_recommend_to_12557 = items_bought_by_12557 - items_bought_by_12431
items_to_recommend_to_12557


# In[25]:


df.loc[df['StockCode'].isin(items_to_recommend_to_12557), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode')


# In[26]:


most_similar_user = user_user_sim_matrix.loc[12557].sort_values(ascending=False).reset_index().iloc[1, 0]
most_similar_user


# In[27]:


def get_items_to_recommend_cust(cust_a): 
  most_similar_user = user_user_sim_matrix.loc[cust_a].sort_values(ascending=False).reset_index().iloc[1, 0]
  items_bought_by_cust_a = set(customer_item_matrix.loc[cust_a].iloc[customer_item_matrix.loc[cust_a].to_numpy().nonzero()].index)
  items_bought_by_cust_b = set(customer_item_matrix.loc[most_similar_user].iloc[customer_item_matrix.loc[most_similar_user].to_numpy().nonzero()].index)
  items_to_recommend_to_a = items_bought_by_cust_b - items_bought_by_cust_a
  items_description = df.loc[df['StockCode'].isin(items_to_recommend_to_a), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode')
  return items_description


# In[28]:


get_items_to_recommend_cust(12557.0)


# In[29]:


get_items_to_recommend_cust(12431.0)


# ## (B) Creating Item to Item similarity matrix

# In[30]:


item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))
item_item_sim_matrix.head()


# In[31]:


item_item_sim_matrix.shape


# In[32]:


item_item_sim_matrix.columns = customer_item_matrix.T.index

item_item_sim_matrix['StockCode'] = customer_item_matrix.T.index
item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')
item_item_sim_matrix.head()


# In[33]:


item_item_sim_matrix.loc['10002'].sort_values(ascending=False)


# In[34]:


top_10_similar_items = list(item_item_sim_matrix.loc['10002'].sort_values(ascending=False).iloc[:10].index)
top_10_similar_items


# In[35]:


df.head()


# In[36]:


df.loc[df['StockCode'] == '90210A']


# In[37]:


df.loc[df['StockCode'] == '90210A'][:1]


# In[38]:


df.loc[df['StockCode'].isin(top_10_similar_items), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode').loc[top_10_similar_items]


# In[39]:


def get_top_similar_items(item):
  top_10_similar_items = list(item_item_sim_matrix.loc[item].sort_values(ascending=False).iloc[:10].index)
  top_10 = df.loc[df['StockCode'].isin(top_10_similar_items), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode').loc[top_10_similar_items]
  return top_10


# In[40]:


get_top_similar_items('84029E')

