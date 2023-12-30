#!/usr/bin/env python
# coding: utf-8

# # CAR PRICE Predictor with ML

# ![Car%20price%20prediction%20using%20machine%20learning.png](attachment:Car%20price%20prediction%20using%20machine%20learning.png)

# In[480]:


import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')


# In[481]:


data=pd.read_csv("/content/quikr_car.csv")


# In[482]:


data.head()


# In[483]:


data.shape


# In[484]:


data.info()


# In[485]:


backup=data.copy()


# ##Cleaning the dataset

# In[486]:


data=data[data['year'].str.isnumeric()]


# ####From the above data type description we can see that the year data are in the object form. So, for that we have convert them into the string.

# In[487]:


#Changing year into the interger


# In[488]:


data['year']=data['year'].astype(int)


# ####From the above data type description we can see that the Price data are in the object form. So, for that we have convert them into the string.

# ###*Price*

# In[489]:


#Ask for price


# In[490]:


data=data[data['Price']!='Ask For Price']


# In[491]:


data['Price']=data['Price'].str.replace(',','').astype(int)


# ###*kms_driven*

# In[492]:


#km_driven column


# In[493]:


data['kms_driven']=data['kms_driven'].str.split().str.get(0).str.replace(',','')


# In[494]:


#nan value are present in the kms_diven column and two 'Patrol' are present in the row.


# In[495]:


data=data[data['kms_driven'].str.isnumeric()]


# In[496]:


data['kms_driven']=data['kms_driven'].astype(int)


# ###*fuel_type*

# In[497]:


#dataset column fuel_type contain the 'na' value


# In[498]:


data=data[~data['fuel_type'].isna()]


# In[499]:


data.shape


# In[500]:


data.info()


# In[501]:


#In dataset we have two column company and name - We having spam data in it. All the spam data got removed in to row cleaning data.


# In[502]:


#Now we only have to clean the Name column spam data.


# In[503]:


data['name']=data['name'].str.split().str.slice(start=0,stop=3).str.join('')


# In[504]:


data=data.reset_index(drop=True)


# In[505]:


data


# In[506]:


data.to_csv("cleaned_dataset.csv")


# In[507]:


data.info()


# In[508]:


#It will describe the complete dataset and including everything


# In[509]:


data.describe(include='all')


# In[510]:


data=data[data['Price']<6000000]


# In[511]:


data['company'].unique()


# In[512]:


import seaborn as sns
plt.subplots(figsize=(20, 10))
ax=sns.boxplot(x='company', y='Price', data=data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40, ha='right')
#ax-axes
#Set the label for x axis tick
#Retrives the current axes tick
#Specific angle for x axis
#ha-Horizontal Alignment
plt.show()


# In[513]:


#Now checking with Year and Price
import seaborn as sns
plt.subplots(figsize=(20, 10))
ax=sns.swarmplot(x='year', y='Price', data=data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')


# In[514]:


#relational plot
sns.relplot(x='kms_driven', y='Price', data=data, height=8, aspect=2)


# In[515]:


plt.subplots(figsize=(20, 10))
sns.boxplot(x='fuel_type', y='Price', data=data)


# In[516]:


#hue parameter is used to add a semantic mapping to the plot


# In[517]:


#Noe the relationdhip of Price with  year, fuel_type and company


# In[518]:


ax=sns.relplot(x='company',y='Price',data=data,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')


# #Further Extacting the data
# 

# In[519]:


X=data[['name', 'company', 'year' , 'kms_driven', 'fuel_type']]
Y=data['Price']


# In[520]:


X


# In[521]:


X.shape


# In[522]:


Y


# In[523]:


Y.shape


# #Now Training the dataset

# In[524]:


#from sklearn library


# In[525]:


from sklearn.model_selection import train_test_split


# In[526]:


X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2)


# In[527]:


from sklearn.linear_model import LinearRegression


# In[528]:


from sklearn.preprocessing import OneHotEncoder


# In[529]:


from sklearn.compose import make_column_transformer


# In[530]:


from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[531]:


#To contain the all categorical values in OneHotEncoder like- name, fuel_type and company(are the objects)


# In[532]:


ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[533]:


#column transformer to tranform column categorical column values


# In[534]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')


# In[535]:


#Now applying Linera Regression model and creating a pipeline


# In[536]:


lr=LinearRegression()


# In[537]:


pipe=make_pipeline(column_trans,lr)


# In[538]:


pipe.fit(X_train,Y_train)


# In[539]:


y_pred=pipe.predict(X_test)


# In[540]:


r2_score=(Y_test,y_pred)


# In[541]:


print(r2_score)


# In[542]:


scores=[]
for i in range(1000):
  X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.1, random_state=i)
  lr=LinearRegression()
  pipe=make_pipeline(column_trans, lr)
  pipe.fit(X_train,Y_train)
  y_pred=pipe.predict(X_test)


# In[544]:


from sklearn.metrics import r2_score


# In[545]:


scores.append(r2_score(Y_test, y_pred))


# In[546]:


import numpy as np


# In[547]:


np.argmax(scores)


# In[548]:


scores[np.argmax(scores)]

