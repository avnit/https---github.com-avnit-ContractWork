
# coding: utf-8

# #  Beta Calculator 
# ## Author : Avnit Bambah
# ### Date : 03/12/2018
# ###### Learning ML with python 3 on Pluralsights.
# ###### Predicting stock beta 

# In[1]:


import pandas as pd
import matplotlib as plt
import numpy as np 

# do ploting inline instead of seperate windows 
get_ipython().magic('matplotlib inline')


# In[3]:


df = pd.read_csv("./holdings-xlk.csv")
df.shape


# In[4]:


df.head(10)


# In[6]:


df.corr()



# In[9]:


import googlefinance


# In[48]:


from pandas_datareader import data
import pandas as pd


# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
tickers = ['AAPL', 'MSFT', 'SPY','CME','GOOG','VVI','agg']

# Define which online source one should use
data_source = 'yahoo'

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2018-01-12'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader(tickers, data_source, start_date, end_date)

#del panel_data["2017-01-02"]

# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data.loc['Adj Close']


close.dropna(axis=0, how='any')

# Getting all weekdays between 01/01/2000 and 12/31/2016
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays).dropna()

close.head(10)


# In[49]:


vr = close.corr()
vr.head(10)


# In[50]:


covar = close.cov()
covar.head(10)


# In[84]:


market_temp = close.iloc[:,4]
market = market_temp.dropna()
count = len(market)
len(market)
market.tail(10)


# In[87]:


total = sum(market)
mean_value = total/count
print(mean_value)
print(market[[0][0]])
variance = ((mean_value - market[[0][0]]) ** 2)
print(variance)
market.head(10)


# In[100]:


covartotal = close.sum()
print(covartotal)
for i in range(1,count-1) :
   # print(covar.iloc[i]['SPY'])
     variance += ((mean_value - market[[i][0]]) ** 2)
print(variance)


# In[110]:


beta_appl = (covartotal/variance) *100  


# In[111]:


beta_appl.head(10)

