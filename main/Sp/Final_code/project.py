#!/usr/bin/env python
# coding: utf-8

# In[83]:


# import necessary library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import yfinance as yf
from datetime import datetime, timedelta


# In[84]:


# income_statement_file = "Profit&Loss.csv"
pfizer_file = 'pfizer_data.csv'
df = pd.read_csv(pfizer_file)

print(f'Original Data Shape: {df.shape}')

# Data cleaning by dropping NaN
df = df.dropna(thresh=1)
df = df.dropna(axis=1)

# Set first column as index
df = df.set_index("Release Date")


# In[85]:


df


# In[86]:


# Removed all columns having all 0s
df = df.drop(df.columns[df.eq(0).all()], axis=1)

# Force data column data to be numeric by removing comma
df = df.replace(',', '', regex=True)
df = df.apply(pd.to_numeric)

print(f'Cleaned Data Shape: {df.shape}')


# In[87]:


# Calculate the correlation matrix
corr_matrix = df.corr().abs()

# Set the threshold
threshold = 0.99

# Identify highly correlated features
high_corr_features = np.where(corr_matrix > threshold)

# Remove highly correlated features
high_corr_features = []
for x, y in zip(*high_corr_features): 
    if x != y and x < y:
        col_x_y = (corr_matrix.columns[x], corr_matrix.columns[y]) 
        high_corr_features.append(col_x_y)
        
for feature in high_corr_features:
    df.drop(feature[1], axis=1 ,inplace=True ,errors='ignore')

print(f'Uncorelated Data Shape: {df.shape}')    


# In[88]:


plt.rcParams['figure.figsize'] = df.shape
sns.heatmap(df.corr())
plt.show()


# In[89]:



def get_last_day_price(symbol, date):
    start_date = datetime.strptime(date, '%d-%b-%y')
    end_date = start_date + timedelta(days=1)
    stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
    last_day_price = stock.iloc[-1]
    price_diff = last_day_price['Close'] - last_day_price['Open']
    return price_diff

def get_price(df, symbol):
    df_index = df.index.to_list()
    price_dict = {}
    for i in df_index:
        price_dict[i] = get_last_day_price(symbol, i)
    y_df = pd.DataFrame.from_dict(price_dict, orient='index', columns=['Price'])
    y_df.index.name = 'Release Date'
    return y_df
    


# In[90]:


symbol = 'PFE' # Pfizer, NYSE
ydf = get_price(df, symbol)

market = '^IXIC' # NASDAQ Composite, NYSE
market = get_price(df, market)

df = pd.concat([df, market], axis=1)


# In[91]:


ydf


# In[92]:


x, y = df, ydf

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test & Train set results
y_test_pred = regressor.predict(x_test)
y_train_pred = regressor.predict(x_train)


# In[93]:


# Visualizing the Training set results
viz_train = plt
viz_train.scatter(x.index, y, color='red')
viz_train.scatter(x_test.index, y_test_pred, color='blue')
viz_train.title('Stock Price VS Date')
viz_train.xlabel('Date')
viz_train.ylabel('Price')
# viz_train.figure(figsize=(8, 4))
viz_train.show() # Note: Image generated are zoomed out, double click to get zoomed in figure


# In[94]:


from sklearn.metrics import r2_score

# Calculate R-squared score for testing set predictions
r2_score(y_test, y_test_pred)


# In[95]:


from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, y_test_pred)


print("MSE (test set): ", mse_test)


# In[ ]:





# In[ ]:




