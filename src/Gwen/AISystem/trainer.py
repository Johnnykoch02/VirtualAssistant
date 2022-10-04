#Imports
import os
from preprocess import *
import subprocess
import tensorflow as tf
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt


try:
    from main import PROJECT_PATH
except:
    pass
import sys



wandb.init()
config = wandb.config

config.max_len = 11
config.buckets = 20

# Save data to array file first
Data_Path = "D:/OneDrive/GwenSmartHome/data/Models/SpeechModel/Training/"
save_data_to_array(max_len=config.max_len,
                   n_mfcc=config.buckets)
X_train, X_test, y_train, y_test = get_train_test()

channels = 1
config.epochs = 5
config.batch_size = 15

# 
num_classes = 2

X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)

plt.imshow(X_train[100, :, :, 0])
print(y_train[100])

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len)

model = Sequential()
model.add(Flatten(input_shape=(config.buckets, config.max_len)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

