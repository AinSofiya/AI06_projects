# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:38:31 2022

@author: AininSofiya

SECOND PROJECT

a- Create a model to predict employees' productivity

b- Link to dataset:
    https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees
    
    
c- Try to achieve similar criteria as Project 1

d- Don't forget to upload to your Github, and make sure your Github has a presentable README


"""

#1. Import packages
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime, os 
import pandas as pd


#2. Read the data from the url file
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00597/garments_worker_productivity.csv')

#%%
print(data.isna().sum())

#%%

#3. Data cleaning
#convert date to datetime
data['date'] = pd.to_datetime(data['date'])
# changing error in sweing
data['department'] = data['department'].apply(lambda x: 'finishing' if x == ('finishing ' or 'finishing' ) else 'sewing' )
#fill missing value
data['wip'].fillna(int(data['wip'].mean()), inplace=True)
#(a) Drop less useful columns
#data = data.drop(['date','day','quarter'],axis=1)


print(data.isnull().sum())

#%%

data = data.copy()


#%%

le = LabelEncoder()
data['department'] = le.fit_transform(data['department'])

#4. Split into features and labels
#label_name = 
features = data[['targeted_productivity', 'team','smv','idle_men', 'no_of_style_change']]
labels = data['actual_productivity']


#%%

#x= data.drop(['targeted_productivity'],axis=1)
#y= data['actual_productivity']


SEED=12345
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.2,random_state=SEED)

#7. Perform data normalization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#%%


#5. Prepare a model for regression problem
model = keras.Sequential()

model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1))
#8. Build NN model
#nIn = x_train.shape[-1]
#nClass = y_train.shape[-1]

#Use functional API
#inputs = keras.Input(shape=(nIn,))
#h1 = layers.Dense(64,activation='relu')
#h2 = layers.Dense(32,activation='relu')
#h3 = layers.Dense(16,activation='relu')

#out = layers.Dense(nClass,activation='softmax')

#x = h1(inputs)
#x = h2(x)
#x = h3(x)
#outputs = out(x)

#model = keras.Model(inputs=inputs,outputs=outputs)
#model.summary()

#%%
#Compile the model
model.compile(optimizer='adam',loss='mse',metrics=['mae'])

#%%

#7. Define your callbacks
base_log_path = r"C:\Users\user\Documents\1_SHRDC\TRAINING\DEEP_LEARNING\CodesExercises\tb_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
es = EarlyStopping(monitor = 'val_loss',patience=20)
tb = TensorBoard(log_dir=log_path)


#Train the model
batch_size=16
epochs=100
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=epochs, callbacks=[es,tb])





