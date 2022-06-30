# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:35:06 2022

@author: AininSofiya

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import applications, layers
import datetime
import pathlib


file_path = r"C:\Users\user\Documents\1_SHRDC\TRAINING\DEEP_LEARNING\CodesExercises\dataset\concrete-cracking"
data_dir = pathlib.Path(file_path)

#%%

#Define batch size and image size
SEED = 12345
BATCH_SIZE = 4
IMG_SIZE = (160,160)

#Create  tensorflow
train_dataset = keras.utils.image_dataset_from_directory(
    data_dir,validation_split=0.2,subset = 'training',seed=SEED, shuffle = True, 
    batch_size = BATCH_SIZE,image_size=IMG_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(
    data_dir,validation_split=0.2,subset='validation',seed=SEED,shuffle=True,
    image_size=IMG_SIZE,batch_size=BATCH_SIZE)

#%%

#Further split validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#%%

#Create prefetch dataset for all 3 splits
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#Data is prepared (mostly..)
#%%

#2. Create data augmentation pipeline
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
#Create a layer for data preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input
#Create the base model by using MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#Apply layer freezing
for layer in base_model.layers[:100]:
    layer.trainable = False

base_model.summary()


#%%

#Create classification layer
class_names = train_dataset.class_names
nClass = len(class_names)

global_avg_pooling = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nClass,activation='softmax')

#%%

#Use functional API to construct the entire model
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = global_avg_pooling(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%

#Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
#%%
#Perform model training
EPOCHS = 10
base_log_path = r"C:\Users\user\Documents\1_SHRDC\TRAINING\DEEP_LEARNING\CodesExercises\tb_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '__project3')
tb = keras.callbacks.TensorBoard(log_dir=log_path)

history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])

#%%
#Deploy model to make prediction
test_loss, test_accuracy = model.evaluate(pf_test)
print('--------------------Test Result----------------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%

image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions,axis=1)


#Plot some predictions
plt.figure(figsize=(10,10))

for i in range(4):
    prediction = class_names[class_predictions[i]]
    label = class_names[label_batch[i]]
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title("Prediction: " + prediction + ", Label: " + label)
    plt.axis('off')

