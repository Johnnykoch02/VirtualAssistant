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
# import wandb
# from wandb.keras import WandbCallback
import matplotlib.pyplot as plt


try:
    from main import PROJECT_PATH
except:
    pass
import sys


max_len = 48
buckets = 36
# C:/Users/John/OneDrive/GwenSmartHome/data/Models/SpeechModel/Training/
# Save data to array file first
Data_Path = "C:/Users/John/OneDrive/GwenSmartHome/data/Models/SpeechModel/Training/"
save_data_to_array()
X_train, X_test, y_train, y_test = get_train_test()

channels = 1
epochs = 5
batch_size = 15

# 
num_classes = 2

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)
for i in range(10):
    print(y_train[i])
    plt.imshow(X_train[i, :, :, 0])


y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0], buckets, max_len)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len)
#shape (n x number of buckets x max_len)
print(X_train.shape)
for i in range(10):
    print(y_train[i])
    plt.imshow(X_train[i, :, :])

import torch as th
import torch.nn as nn

class KeywordAudioModel(nn.Module):
    '''
        This Network is designed to spot KeyWords in speech recognition.
        https://arxiv.org/pdf/2005.06720v2.pdf
        The structure of the network is designed to implement the following Paper.
    '''
    def __init__(self, input_shape):
        super(KeywordAudioModel, self).__init__()
        '''3x36x48'''
        self.layers = nn.Sequential(
            nn.BatchNorm2D(input_shape)
            nn.Conv2D(in_channels=3, out_channels=)
        )