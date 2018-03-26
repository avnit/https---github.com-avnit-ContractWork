
#  Beta Calculator 
## Author : Avnit Bambah
### Date : 03/12/2018
###### Learning ML with python 3 on Pluralsights.
###### Predicting stock beta 


```python
import pandas as pd
import matplotlib as plt
import numpy as np 

# do ploting inline instead of seperate windows 
%matplotlib inline
```


```python
# add new stocks to the csv file 
df = pd.read_csv("./holdings-xlk.csv")
df.shape

```




    (20, 2)




```python
# check if there are any stocks in the csv file 
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Company Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ANRI</td>
      <td>Amira Nature Food Ltd</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BETR</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BUFF</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CALM</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CENT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DTEA</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FRPT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KHC</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LANC</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LWAY</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index in addition to the one in the csv file 
tickers = ['AAPL', 'MSFT', 'SPY','CME','GOOG','VVI','agg']
# get the symbols and add them to the list 
symbols = df.iloc[:,[0]]
array = symbols.values.tolist()
for i in range(1 , len(array)):
    tickers.append(str(array[i]).replace('[\'', '').replace('\']',''))
    
tickers 
```




    ['AAPL',
     'MSFT',
     'SPY',
     'CME',
     'GOOG',
     'VVI',
     'agg',
     'BETR',
     'BUFF',
     'CALM',
     'CENT',
     'DTEA',
     'FRPT',
     'KHC',
     'LANC',
     'LWAY',
     'NUTR',
     'PF',
     'POST',
     'PPC',
     'RELV',
     'RIBT',
     'SAFM',
     'TOF',
     'WILC',
     'WWAV']




```python
import googlefinance
```


```python
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

close.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>BETR</th>
      <th>BUFF</th>
      <th>CALM</th>
      <th>CENT</th>
      <th>CME</th>
      <th>DTEA</th>
      <th>FRPT</th>
      <th>GOOG</th>
      <th>KHC</th>
      <th>...</th>
      <th>PPC</th>
      <th>RELV</th>
      <th>RIBT</th>
      <th>SAFM</th>
      <th>SPY</th>
      <th>TOF</th>
      <th>VVI</th>
      <th>WILC</th>
      <th>WWAV</th>
      <th>agg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>2.706315</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.637909</td>
      <td>3.413309</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.883336</td>
      <td>4.125650</td>
      <td>NaN</td>
      <td>4.245998</td>
      <td>103.393242</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>2.478144</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.604914</td>
      <td>3.286890</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.446455</td>
      <td>4.250875</td>
      <td>NaN</td>
      <td>4.003366</td>
      <td>99.349930</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>2.514410</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.604914</td>
      <td>3.265820</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.504705</td>
      <td>4.000824</td>
      <td>NaN</td>
      <td>4.367311</td>
      <td>99.527641</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>2.296816</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.626911</td>
      <td>3.286890</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.825084</td>
      <td>4.000824</td>
      <td>NaN</td>
      <td>4.124682</td>
      <td>97.928093</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>2.405613</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.626911</td>
      <td>3.286890</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.970712</td>
      <td>3.875598</td>
      <td>NaN</td>
      <td>4.245998</td>
      <td>103.615379</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>2.363304</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.626911</td>
      <td>3.202610</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.504705</td>
      <td>3.875598</td>
      <td>NaN</td>
      <td>4.124682</td>
      <td>103.970871</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-11</th>
      <td>2.242418</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615912</td>
      <td>3.244750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.533829</td>
      <td>4.250875</td>
      <td>NaN</td>
      <td>4.245998</td>
      <td>102.726746</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-12</th>
      <td>2.107934</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615912</td>
      <td>3.329029</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.533829</td>
      <td>5.251082</td>
      <td>NaN</td>
      <td>3.912383</td>
      <td>101.704826</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-13</th>
      <td>2.339126</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615912</td>
      <td>3.329029</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.970712</td>
      <td>5.001029</td>
      <td>NaN</td>
      <td>3.980611</td>
      <td>103.082207</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-14</th>
      <td>2.428280</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.621403</td>
      <td>3.202610</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>6.087214</td>
      <td>5.625959</td>
      <td>NaN</td>
      <td>4.306655</td>
      <td>104.481773</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 26 columns</p>
</div>




```python
vr = close.corr()
vr.head(10)

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>BETR</th>
      <th>BUFF</th>
      <th>CALM</th>
      <th>CENT</th>
      <th>CME</th>
      <th>DTEA</th>
      <th>FRPT</th>
      <th>GOOG</th>
      <th>KHC</th>
      <th>...</th>
      <th>PPC</th>
      <th>RELV</th>
      <th>RIBT</th>
      <th>SAFM</th>
      <th>SPY</th>
      <th>TOF</th>
      <th>VVI</th>
      <th>WILC</th>
      <th>WWAV</th>
      <th>agg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>1.000000</td>
      <td>-0.669186</td>
      <td>0.650407</td>
      <td>0.936062</td>
      <td>0.684971</td>
      <td>0.915372</td>
      <td>-0.703757</td>
      <td>0.593848</td>
      <td>0.956831</td>
      <td>0.185599</td>
      <td>...</td>
      <td>0.443312</td>
      <td>-0.486193</td>
      <td>-0.519303</td>
      <td>0.937320</td>
      <td>0.965438</td>
      <td>NaN</td>
      <td>0.708511</td>
      <td>-0.493247</td>
      <td>NaN</td>
      <td>0.887378</td>
    </tr>
    <tr>
      <th>BETR</th>
      <td>-0.669186</td>
      <td>1.000000</td>
      <td>-0.078369</td>
      <td>0.308227</td>
      <td>-0.581386</td>
      <td>-0.588406</td>
      <td>0.747871</td>
      <td>-0.505143</td>
      <td>-0.555503</td>
      <td>-0.119124</td>
      <td>...</td>
      <td>-0.417232</td>
      <td>-0.247830</td>
      <td>0.410661</td>
      <td>-0.637428</td>
      <td>-0.556837</td>
      <td>NaN</td>
      <td>-0.664139</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.140608</td>
    </tr>
    <tr>
      <th>BUFF</th>
      <td>0.650407</td>
      <td>-0.078369</td>
      <td>1.000000</td>
      <td>-0.248258</td>
      <td>0.635458</td>
      <td>0.797233</td>
      <td>-0.531502</td>
      <td>0.716681</td>
      <td>0.698244</td>
      <td>0.069239</td>
      <td>...</td>
      <td>0.598517</td>
      <td>-0.015995</td>
      <td>-0.159265</td>
      <td>0.587096</td>
      <td>0.758926</td>
      <td>NaN</td>
      <td>0.631657</td>
      <td>-0.939039</td>
      <td>NaN</td>
      <td>0.502931</td>
    </tr>
    <tr>
      <th>CALM</th>
      <td>0.936062</td>
      <td>0.308227</td>
      <td>-0.248258</td>
      <td>1.000000</td>
      <td>0.544791</td>
      <td>0.832050</td>
      <td>0.289356</td>
      <td>-0.265112</td>
      <td>0.887796</td>
      <td>-0.731628</td>
      <td>...</td>
      <td>0.407775</td>
      <td>-0.467830</td>
      <td>-0.547360</td>
      <td>0.878630</td>
      <td>0.906301</td>
      <td>NaN</td>
      <td>0.548055</td>
      <td>-0.270749</td>
      <td>NaN</td>
      <td>0.878670</td>
    </tr>
    <tr>
      <th>CENT</th>
      <td>0.684971</td>
      <td>-0.581386</td>
      <td>0.635458</td>
      <td>0.544791</td>
      <td>1.000000</td>
      <td>0.736442</td>
      <td>-0.835450</td>
      <td>0.045755</td>
      <td>0.690549</td>
      <td>0.527623</td>
      <td>...</td>
      <td>0.604939</td>
      <td>0.022598</td>
      <td>-0.084792</td>
      <td>0.753446</td>
      <td>0.703491</td>
      <td>NaN</td>
      <td>0.865030</td>
      <td>-0.485567</td>
      <td>NaN</td>
      <td>0.348493</td>
    </tr>
    <tr>
      <th>CME</th>
      <td>0.915372</td>
      <td>-0.588406</td>
      <td>0.797233</td>
      <td>0.832050</td>
      <td>0.736442</td>
      <td>1.000000</td>
      <td>-0.813358</td>
      <td>0.301428</td>
      <td>0.958832</td>
      <td>0.221677</td>
      <td>...</td>
      <td>0.515731</td>
      <td>-0.451223</td>
      <td>-0.287172</td>
      <td>0.900730</td>
      <td>0.957320</td>
      <td>NaN</td>
      <td>0.886091</td>
      <td>-0.413933</td>
      <td>NaN</td>
      <td>0.748169</td>
    </tr>
    <tr>
      <th>DTEA</th>
      <td>-0.703757</td>
      <td>0.747871</td>
      <td>-0.531502</td>
      <td>0.289356</td>
      <td>-0.835450</td>
      <td>-0.813358</td>
      <td>1.000000</td>
      <td>-0.057489</td>
      <td>-0.858145</td>
      <td>-0.300839</td>
      <td>...</td>
      <td>-0.474996</td>
      <td>0.288723</td>
      <td>0.624573</td>
      <td>-0.798169</td>
      <td>-0.806289</td>
      <td>NaN</td>
      <td>-0.846973</td>
      <td>0.249263</td>
      <td>NaN</td>
      <td>-0.655630</td>
    </tr>
    <tr>
      <th>FRPT</th>
      <td>0.593848</td>
      <td>-0.505143</td>
      <td>0.716681</td>
      <td>-0.265112</td>
      <td>0.045755</td>
      <td>0.301428</td>
      <td>-0.057489</td>
      <td>1.000000</td>
      <td>0.061231</td>
      <td>0.072662</td>
      <td>...</td>
      <td>0.473951</td>
      <td>0.602256</td>
      <td>0.470791</td>
      <td>0.345904</td>
      <td>0.393145</td>
      <td>NaN</td>
      <td>0.221741</td>
      <td>-0.376312</td>
      <td>NaN</td>
      <td>-0.035979</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>0.956831</td>
      <td>-0.555503</td>
      <td>0.698244</td>
      <td>0.887796</td>
      <td>0.690549</td>
      <td>0.958832</td>
      <td>-0.858145</td>
      <td>0.061231</td>
      <td>1.000000</td>
      <td>0.244621</td>
      <td>...</td>
      <td>0.400323</td>
      <td>-0.627298</td>
      <td>-0.405371</td>
      <td>0.940656</td>
      <td>0.971929</td>
      <td>NaN</td>
      <td>0.790159</td>
      <td>-0.402471</td>
      <td>NaN</td>
      <td>0.836116</td>
    </tr>
    <tr>
      <th>KHC</th>
      <td>0.185599</td>
      <td>-0.119124</td>
      <td>0.069239</td>
      <td>-0.731628</td>
      <td>0.527623</td>
      <td>0.221677</td>
      <td>-0.300839</td>
      <td>0.072662</td>
      <td>0.244621</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.200496</td>
      <td>0.084573</td>
      <td>-0.690083</td>
      <td>0.298551</td>
      <td>0.312467</td>
      <td>NaN</td>
      <td>0.360078</td>
      <td>-0.385210</td>
      <td>NaN</td>
      <td>0.605116</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 26 columns</p>
</div>




```python
covar = close.cov()
covar.head(10)

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>BETR</th>
      <th>BUFF</th>
      <th>CALM</th>
      <th>CENT</th>
      <th>CME</th>
      <th>DTEA</th>
      <th>FRPT</th>
      <th>GOOG</th>
      <th>KHC</th>
      <th>...</th>
      <th>PPC</th>
      <th>RELV</th>
      <th>RIBT</th>
      <th>SAFM</th>
      <th>SPY</th>
      <th>TOF</th>
      <th>VVI</th>
      <th>WILC</th>
      <th>WWAV</th>
      <th>agg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>2222.202734</td>
      <td>-51.143884</td>
      <td>86.247963</td>
      <td>694.738568</td>
      <td>265.995337</td>
      <td>1466.740664</td>
      <td>-82.765683</td>
      <td>67.761889</td>
      <td>11591.169170</td>
      <td>37.728778</td>
      <td>...</td>
      <td>179.671918</td>
      <td>-481.554740</td>
      <td>-3827.297967</td>
      <td>1435.955249</td>
      <td>2337.901203</td>
      <td>NaN</td>
      <td>352.069745</td>
      <td>-0.355950</td>
      <td>NaN</td>
      <td>652.981952</td>
    </tr>
    <tr>
      <th>BETR</th>
      <td>-51.143884</td>
      <td>8.469221</td>
      <td>-0.874500</td>
      <td>5.093134</td>
      <td>-15.792594</td>
      <td>-35.436553</td>
      <td>8.089958</td>
      <td>-5.467575</td>
      <td>-206.356633</td>
      <td>-2.486708</td>
      <td>...</td>
      <td>-5.513968</td>
      <td>-1.072438</td>
      <td>0.567604</td>
      <td>-52.470184</td>
      <td>-41.257740</td>
      <td>NaN</td>
      <td>-20.738042</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.847770</td>
    </tr>
    <tr>
      <th>BUFF</th>
      <td>86.247963</td>
      <td>-0.874500</td>
      <td>23.115937</td>
      <td>-6.656368</td>
      <td>29.778206</td>
      <td>92.232768</td>
      <td>-10.094158</td>
      <td>13.691463</td>
      <td>472.453186</td>
      <td>2.471830</td>
      <td>...</td>
      <td>12.812706</td>
      <td>-0.114160</td>
      <td>-0.389499</td>
      <td>79.594885</td>
      <td>99.745831</td>
      <td>NaN</td>
      <td>33.249147</td>
      <td>-0.010075</td>
      <td>NaN</td>
      <td>4.958275</td>
    </tr>
    <tr>
      <th>CALM</th>
      <td>694.738568</td>
      <td>5.093134</td>
      <td>-6.656368</td>
      <td>247.884557</td>
      <td>70.658497</td>
      <td>433.381507</td>
      <td>8.014144</td>
      <td>-7.254609</td>
      <td>3473.405906</td>
      <td>-30.441695</td>
      <td>...</td>
      <td>55.198158</td>
      <td>-154.759782</td>
      <td>-1297.838754</td>
      <td>449.564252</td>
      <td>733.005508</td>
      <td>NaN</td>
      <td>87.953866</td>
      <td>-0.014825</td>
      <td>NaN</td>
      <td>208.034860</td>
    </tr>
    <tr>
      <th>CENT</th>
      <td>265.995337</td>
      <td>-15.792594</td>
      <td>29.778206</td>
      <td>70.658497</td>
      <td>67.860877</td>
      <td>198.623289</td>
      <td>-44.356682</td>
      <td>2.332354</td>
      <td>1471.136222</td>
      <td>38.850326</td>
      <td>...</td>
      <td>42.845016</td>
      <td>3.911320</td>
      <td>-106.745066</td>
      <td>201.707986</td>
      <td>297.699522</td>
      <td>NaN</td>
      <td>75.540931</td>
      <td>-0.034895</td>
      <td>NaN</td>
      <td>43.795155</td>
    </tr>
    <tr>
      <th>CME</th>
      <td>1466.740664</td>
      <td>-35.436553</td>
      <td>92.232768</td>
      <td>433.381507</td>
      <td>198.623289</td>
      <td>1107.668606</td>
      <td>-95.293370</td>
      <td>33.283480</td>
      <td>7555.474566</td>
      <td>39.594868</td>
      <td>...</td>
      <td>151.502388</td>
      <td>-324.728354</td>
      <td>-1421.099020</td>
      <td>923.006151</td>
      <td>1685.512973</td>
      <td>NaN</td>
      <td>286.588685</td>
      <td>-0.190850</td>
      <td>NaN</td>
      <td>369.772985</td>
    </tr>
    <tr>
      <th>DTEA</th>
      <td>-82.765683</td>
      <td>8.089958</td>
      <td>-10.094158</td>
      <td>8.014144</td>
      <td>-44.356682</td>
      <td>-95.293370</td>
      <td>21.061896</td>
      <td>-1.226686</td>
      <td>-690.122405</td>
      <td>-9.310695</td>
      <td>...</td>
      <td>-8.525583</td>
      <td>2.428024</td>
      <td>3.911218</td>
      <td>-95.315793</td>
      <td>-100.203509</td>
      <td>NaN</td>
      <td>-46.097047</td>
      <td>0.002325</td>
      <td>NaN</td>
      <td>-7.783643</td>
    </tr>
    <tr>
      <th>FRPT</th>
      <td>67.761889</td>
      <td>-5.467575</td>
      <td>13.691463</td>
      <td>-7.254609</td>
      <td>2.332354</td>
      <td>33.283480</td>
      <td>-1.226686</td>
      <td>20.528429</td>
      <td>48.038368</td>
      <td>2.170361</td>
      <td>...</td>
      <td>8.729568</td>
      <td>4.579280</td>
      <td>2.332316</td>
      <td>42.346433</td>
      <td>47.337300</td>
      <td>NaN</td>
      <td>11.464260</td>
      <td>-0.036775</td>
      <td>NaN</td>
      <td>-0.378159</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>11591.169170</td>
      <td>-206.356633</td>
      <td>472.453186</td>
      <td>3473.405906</td>
      <td>1471.136222</td>
      <td>7555.474566</td>
      <td>-690.122405</td>
      <td>48.038368</td>
      <td>63383.871446</td>
      <td>261.315275</td>
      <td>...</td>
      <td>914.203627</td>
      <td>-3501.813767</td>
      <td>-15594.065694</td>
      <td>7214.066331</td>
      <td>12747.534615</td>
      <td>NaN</td>
      <td>2054.867736</td>
      <td>-2.295279</td>
      <td>NaN</td>
      <td>3055.911118</td>
    </tr>
    <tr>
      <th>KHC</th>
      <td>37.728778</td>
      <td>-2.486708</td>
      <td>2.471830</td>
      <td>-30.441695</td>
      <td>38.850326</td>
      <td>39.594868</td>
      <td>-9.310695</td>
      <td>2.170361</td>
      <td>261.315275</td>
      <td>55.226767</td>
      <td>...</td>
      <td>6.608097</td>
      <td>0.962938</td>
      <td>-2.866791</td>
      <td>62.786642</td>
      <td>63.315429</td>
      <td>NaN</td>
      <td>29.476574</td>
      <td>-0.105540</td>
      <td>NaN</td>
      <td>9.452360</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 26 columns</p>
</div>




```python
market_temp = close.iloc[:,[20]]
market = market_temp.dropna()

```

    SPY    571340.017387
    dtype: float64



```python
#total = sum(market)
mean_value = pd.DataFrame.mean(market)
print(mean_value)
variance = pd.DataFrame.var(market)
print(variance)
market.head(10)
```

    SPY    124.610691
    dtype: float64
    SPY    2638.883322
    dtype: float64





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SPY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>103.393242</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>99.349930</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>99.527641</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>97.928093</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>103.615379</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>103.970871</td>
    </tr>
    <tr>
      <th>2000-01-11</th>
      <td>102.726746</td>
    </tr>
    <tr>
      <th>2000-01-12</th>
      <td>101.704826</td>
    </tr>
    <tr>
      <th>2000-01-13</th>
      <td>103.082207</td>
    </tr>
    <tr>
      <th>2000-01-14</th>
      <td>104.481773</td>
    </tr>
  </tbody>
</table>
</div>




```python
covartotal = pd.DataFrame.sum(covar)
print(covartotal)
print(variance[[0][0]])

```

    AAPL     20183.781907
    BETR      -459.850045
    BUFF      1037.536241
    CALM      5933.322620
    CENT      3262.749249
    CME      14873.458114
    DTEA     -1334.633658
    FRPT       269.015945
    GOOG    112733.001666
    KHC        741.789951
    LANC     13850.834851
    LWAY       785.685664
    MSFT      7454.773054
    NUTR         0.000000
    PF        4623.434309
    POST      7835.332044
    PPC       2734.003288
    RELV     -2898.545207
    RIBT     -6549.491353
    SAFM     13439.552800
    SPY      23507.262977
    TOF          0.000000
    VVI       4551.484670
    WILC        -4.062500
    WWAV         0.000000
    agg       4596.941615
    dtype: float64
    2638.88332232



```python
beta_appl = (covartotal/variance[[0][0]])

```


```python
beta_appl.head(100)
```




    AAPL     7.648607
    BETR    -0.174259
    BUFF     0.393172
    CALM     2.248422
    CENT     1.236413
    CME      5.636270
    DTEA    -0.505757
    FRPT     0.101943
    GOOG    42.719964
    KHC      0.281100
    LANC     5.248748
    LWAY     0.297734
    MSFT     2.824973
    NUTR     0.000000
    PF       1.752042
    POST     2.969185
    PPC      1.036046
    RELV    -1.098398
    RIBT    -2.481918
    SAFM     5.092894
    SPY      8.908034
    TOF      0.000000
    VVI      1.724777
    WILC    -0.001539
    WWAV     0.000000
    agg      1.742003
    dtype: float64


