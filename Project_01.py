#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


from utils.preprocess import preprocess_dataset
from utils.EDA import histogram, bivariate_analysis, get_correlation
from utils.MLP_model import split_data, build_model, compile_model, train_model, get_accuracy, predict, con_matrix, roc_curve_draw
import pandas as pd


# # Import dataset

# In[1]:


client_data = pd.read_csv('Data/default_of_credit_card_clients.csv')
client_data.drop('ID', inplace=True, axis=1)
client_data.head()


# In[3]:


client_data.rename(columns={"default payment next month": "target"}, inplace=True)
client_data.head()


# # Data Preprocessing

# In[4]:


preproccessed_data = preprocess_dataset(client_data)


# # Exploratory Data Analysis

# In[5]:


histogram(preproccessed_data)


# In[6]:


bivariate_analysis(preproccessed_data)


# In[7]:


get_correlation(client_data)


# # Building the MLP Model

# In[ ]:


x_train, x_test, y_train, y_test, x_val, y_val = split_data(preproccessed_data)


# In[ ]:


print("Feature matrix:", x_train.shape)
print("Target matrix:", y_train.shape)
print("Feature matrix:", x_test.shape)
print("Target matrix:", y_test.shape)
print("X_Validation matrix:", x_val.shape)
print("y_validation matrix:", y_val.shape)


# In[ ]:


model = build_model()


# In[ ]:


compile_model(model)


# In[ ]:


model = train_model(model, x_train, y_train)


# In[ ]:


get_accuracy(model, x_test, y_test)


# In[ ]:


y_pred = predict(model, x_test)


# In[ ]:


con_matrix(y_test, y_pred)


# In[ ]:


roc_curve_draw(model, x_test, y_test)

