
# coding: utf-8

# #  Beta Calculator 
# ## Author : Avnit Bambah
# ### Date : 03/12/2018
# ###### Learning ML with python 3 on Pluralsights.
# ###### Predicting stock beta 

# In[250]:


import pandas as pd
import matplotlib as plt
import numpy as np 

# do ploting inline instead of seperate windows 
get_ipython().magic('matplotlib inline')


# In[251]:


# add new stocks to the csv file 
df = pd.read_csv("./holdings-xlk.csv")
df.shape


# In[252]:


# check if there are any stocks in the csv file 
df.head(10)


# In[253]:


# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index in addition to the one in the csv file 
tickers = ['AAPL', 'MSFT', 'SPY','CME','GOOG','VVI','agg']
# get the symbols and add them to the list 
symbols = df.iloc[:,[0]]
array = symbols.values.tolist()
for i in range(1 , len(array)):
    tickers.append(str(array[i]).replace('[\'', '').replace('\']',''))
    
tickers 


# In[254]:


import googlefinance


# In[259]:


from pandas_datareader import data
import pandas as pd

# Define which online source one should use
data_source = 'yahoo'

# We would like all available data from 01/01/2000 until 12/31/2018.
start_date = '2000-01-01'
end_date = '2018-03-24'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
try:
    panel_data = data.DataReader(tickers, data_source, start_date, end_date)
except:
    print('error in the symbol')
#del panel_data["2017-01-02"]

# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data.loc['Adj Close']

close.dropna(axis=0, how='any')

# Getting all weekdays between start date and end date 
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

close_onedayold = close[:-1]

close_onedayold.head(10)

close_final = close - close_onedayold


# In[257]:


vr = close.corr()
vr.head(10)


# In[258]:


covar = close_final.cov()
covar.head(10)


# In[223]:


market_temp = close_final.iloc[:,[20]]
market = market_temp.dropna()


# In[224]:


#total = sum(market)
mean_value = pd.DataFrame.mean(market)
print(mean_value)
variance = pd.DataFrame.var(market)
print(variance)
market.head(10)


# In[239]:


covartotal = pd.DataFrame.sum(covar)
print(covartotal)
print(variance[[0][0]])


# In[242]:


beta_appl = (covartotal/variance[[0][0]])


# In[243]:


beta_appl.head(100)


# In[247]:


import matplotlib.pyplot as plt
plt.plot(beta_appl)

