#!/usr/bin/env python
# coding: utf-8

# # Data loading

# In[1]:


import pandas as pd


# In[2]:


control_data = pd.read_csv('control_data.csv')
experiment_data = pd.read_csv('experiment_data.csv')


# In[3]:


control_data.head()


# In[4]:


experiment_data.head()


# In[5]:


control_data.info()


# In[6]:


experiment_data.info()


# In[7]:


control_data.isna().sum()


# In[8]:


experiment_data.isna().sum()


# In[9]:


control_data[control_data['Enrollments'].isna()]


# # Data wrangling

# In[10]:


# Combine with Experiment data
data_total = pd.concat([control_data, experiment_data])
data_total.sample(10)


# In[11]:


import numpy as np
np.random.seed(7)
import sklearn.utils


# In[12]:


# Add row id
data_total['row_id'] = data_total.index


# In[13]:


# Create a Day of Week feature
data_total['DOW'] = data_total['Date'].str.slice(start=0, stop=3)


# In[14]:


# Remove missing data
data_total.dropna(inplace=True)


# In[15]:


# Add a binary column Experiment to denote
# if the data was part of the experiment or not (Random)
data_total['Experiment'] = np.random.randint(2, size=len(data_total))


# In[16]:


# Remove missing data
data_total.dropna(inplace=True)

# Remove Date and Payments columns
del data_total['Date'], data_total['Payments']

# Shuffle the data
data_total = sklearn.utils.shuffle(data_total)


# In[17]:


# Check the new data
data_total.head()


# In[18]:


# Reorder the columns 
data_total = data_total[['row_id', 'Experiment', 'Pageviews', 'Clicks', 'DOW', 'Enrollments']]


# In[19]:


# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_total.loc[:, data_total.columns != 'Enrollments'],\
                                                    data_total['Enrollments'], test_size=0.2)


# In[20]:


# Converting strings to numbers
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
X_train['DOW'] = lb.fit_transform(X_train['DOW'])
X_test['DOW'] = lb.transform(X_test['DOW'])


# In[21]:


X_train.head()


# In[22]:


X_test.head()


# ### Helper functions
# - Function for printing the evaluation scores related to a regression problem
# - Function for plotting the original values and values predicted by the model

# In[23]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))


# In[24]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_preds(y_test, y_preds, model_name):
    N = len(y_test)
    plt.figure(figsize=(10,5))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Oberservation')
    plt.ylabel('Enrollments')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()


# # Linear regression:

# In[26]:


get_ipython().system(' pip install statsmodels')


# In[27]:


import statsmodels.api as sm

X_train_refined = X_train.drop(columns=['row_id'], axis=1)
linear_regression = sm.OLS(y_train, X_train_refined)
linear_regression = linear_regression.fit()


# In[28]:


X_test_refined = X_test.drop(columns=['row_id'], axis=1)
y_preds = linear_regression.predict(X_test_refined)


# In[29]:


calculate_metrics(y_test, y_preds)


# In[30]:


plot_preds(y_test, y_preds, 'Linear Regression')


# In[31]:


print(linear_regression.summary())


# In[32]:


pd.DataFrame(linear_regression.pvalues)\
    .reset_index()\
    .rename(columns={'index':'Terms', 0:'p_value'})\
    .sort_values('p_value')


# # Model 02: Decision Tree

# In[33]:


from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf =4, random_state=7)
dtree.fit(X_train_refined, y_train)
y_preds = dtree.predict(X_test_refined)

calculate_metrics(y_test, y_preds)


# In[34]:


plot_preds(y_test, y_preds, 'Decision Tree')


# # Model 03: XGBoost

# In[45]:


get_ipython().system(' pip install xgboost')


# In[46]:


import xgboost as xgb


# In[47]:


DM_train = xgb.DMatrix(data=X_train_refined,label=y_train)
DM_test = xgb.DMatrix(data=X_test_refined,label=y_test)


# In[51]:


parameters = {
    'max_depth': 6,
    'objective': 'reg:squarederror',
    'eta': 0.2,  
    'lambda': 0.01, 
}

num_boost_round = 1000  # Number of boosting rounds (previously 'n_estimators')

xg_reg = xgb.train(params=parameters, dtrain=DM_train, num_boost_round=num_boost_round)
y_preds = xg_reg.predict(DM_test)


# In[52]:


# Assuming you have a function 'calculate_metrics()' to evaluate your model's performance
calculate_metrics(y_test, y_preds)


# In[53]:


plot_preds(y_test, y_preds, 'XGBoost')


# # H2O.ai's AutoML

# In[55]:


get_ipython().system(' pip install h2o')


# In[56]:


import h2o
from h2o.automl import H2OAutoML
h2o.init()


# In[57]:


X_train['Enrollments'] = y_train
X_test['Enrollments'] = y_test


# In[58]:


X_train_h2o = h2o.H2OFrame(X_train)
X_test_h2o = h2o.H2OFrame(X_test)


# In[59]:


features = X_train.columns.values.tolist()
target = "Enrollments"


# In[60]:


# Construct the AutoML pipeline
auto_h2o = H2OAutoML()
# Train 
auto_h2o.train(x=features,
               y=target,
               training_frame=X_train_h2o)


# In[61]:


calculate_metrics(y_test, y_preds) # XGBoost is effiecient among them 


# In[ ]:




