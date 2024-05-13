#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.chdir("C:\\Users\\souvick\\Downloads")


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

train_data = pd.read_csv('train.csv')

features = train_data.drop(['candidate_id', 'triglyceride_lvl'], axis=1)
target = train_data['triglyceride_lvl']

features = pd.get_dummies(features)

features = features.fillna(features.mean())

features = features.replace([np.inf], np.finfo('float64').max)

features = features.replace([-np.inf], np.finfo('float64').min)

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)

score = max(0, 100 * mae)

print(f'Score: {score}')

test_data = pd.read_csv('test.csv')

test_features = test_data.drop(['candidate_id'], axis=1)

test_features = pd.get_dummies(test_features)

test_features = test_features.fillna(test_features.mean())

test_features = test_features.replace([np.inf], np.finfo('float64').max)

test_features = test_features.replace([-np.inf], np.finfo('float64').min)

test_predictions = model.predict(test_features)

submission = pd.DataFrame({
    'candidate_id': test_data['candidate_id'],
    'triglyceride_lvl': test_predictions
})

submission.to_csv('submission.csv', index=False)


# In[ ]:




