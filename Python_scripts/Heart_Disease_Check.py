#!/usr/bin/env python
# coding: utf-8

# # Import Directories

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.metrics import accuracy_score


# # Data Imports

# In[2]:


column_names = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','output']
url='/Users/msr89/Documents/Python_datascience/TensorFlow/Deep_learning/Heart_Disease_pred/input_file/heart.csv'
raw_dataset = pd.read_csv(url, names=column_names)
dataset = raw_dataset.copy()
dataset.describe()


# # Data Visualization

# In[3]:


# Age vs Heart attack chances


sns.distplot(dataset[dataset['output'] == 0]["age"], color='green',kde=True,) 
sns.distplot(dataset[dataset['output'] == 1]["age"], color='red',kde=True)
plt.title('Attack versus Age')
plt.show()

# heart attack gets distributed between age from 40 to 70


# In[4]:


# Cholestrol Vs Attack possibility

sns.distplot(dataset[dataset['output'] == 0]["chol"], color='green',kde=True,) 
sns.distplot(dataset[dataset['output'] == 1]["chol"], color='red',kde=True)
plt.title('Attack versus Cholestrol distribution')
plt.show()

# the attack possiblities are for the people having cholestrol ranges from 150 to 350


# In[20]:


#Resting Blood Pressure and Attack possibilities


sns.distplot(dataset[dataset['output'] == 0]["trtbps"], color='green',kde=True,) 
sns.distplot(dataset[dataset['output'] == 1]["trtbps"], color='red',kde=True)
plt.title('Attack versus Cholestrol distribution')
plt.show()

# occurence of heart attack is high for the resting bp between 100 and 160


# # Split train and Test Data

# In[6]:


train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('age')
test_labels = test_features.pop('age')

print(train_labels.describe())


# In[ ]:





# # Normalize data

# In[7]:


normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())


# # One variable Model

# In[8]:


agechk = np.array(train_features['trtbps'])
agechk_normalizer = preprocessing.Normalization(input_shape=[1,])
agechk_normalizer.adapt(agechk)
agechk_model = tf.keras.Sequential([
   agechk_normalizer,
     layers.Dense(units=1)
])
agechk_model.summary()
agechk_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# # Regression Linear
# 

# In[9]:


history = agechk_model.fit(
    train_features['trtbps'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [agechk]')
  plt.legend()
  plt.grid(True)

plot_loss(history)


# In[10]:


test_results = {}

test_results['agechk_model'] = agechk_model.evaluate(
    test_features['trtbps'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 180, 181)
y = agechk_model.predict(x)


# # Predict age related blood pressure using Linear Regression Keras
# 

# In[11]:


def plot_trtbps(x, y):
  plt.scatter(train_features['trtbps'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('blood pressuge hg')
  plt.ylabel('age')
  plt.legend()
plot_trtbps(x,y)


# # DNN Regression Keras With Single variable Age

# In[12]:


# Build a Model with Normalized input

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


# In[13]:


bpscheck = np.array(train_features['trtbps'])
bpscheck_normalizer = preprocessing.Normalization(input_shape=[1,])
bpscheck_normalizer.adapt(bpscheck)
dnn_bpchk_model = build_and_compile_model(bpscheck_normalizer)


# In[14]:


history = dnn_bpchk_model.fit(
    train_features['trtbps'], train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
plot_loss(history)


# # Predict and Plot DNN SIngle input model

# In[15]:


x_dnn = tf.linspace(0.0, 180, 181)
y_dnn = dnn_bpchk_model.predict(x)

plot_trtbps(x_dnn,y_dnn)


# In[ ]:




