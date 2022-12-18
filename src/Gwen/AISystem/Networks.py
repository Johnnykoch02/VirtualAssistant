#Imports
import os
from preprocess import *
import subprocess
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader





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
    def __init__(self, input_shape, lr=0.015):
        super(KeywordAudioModel, self).__init__()
        '''3x64x48'''
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_shape),
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5,7), padding=5, bias=True), # output (5 x 78 x 68)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4,4), stride=2), # Output (10 x 38 x 33)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(6,5), stride=4), # Output (20 x 9 x 8)
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1440, 2600),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Linear(2600, 900, bias=True), 
            nn.Sigmoid(),
            nn.Linear(900, 120),
            nn.Sigmoid(),
            nn.Linear(120, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self._loss_func = nn.CrossEntropyLoss()
        self._optimizer = th.optim.Adam(self.parameters, lr=lr)
        
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def train(self, num_epochs, data_loader: DataLoader):
        from torch.autograd import Variable
        
        total_steps = len(data_loader['train'])
          
        for epoch in range(num_epochs):
            for i, (mel_imgs, output) in enumerate(data_loader['train']):
                batch_x = Variable(mel_imgs)
                batch_y = Variable(output)
                
                output = self.forward(batch_x)[0]
                
                
                  
        