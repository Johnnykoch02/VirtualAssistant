#Imports
import os
from src.Gwen.AISystem.preprocess import *
import subprocess
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

class Dataset(th.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

max_len = 48
buckets = 36
# C:/Users/John/OneDrive/GwenSmartHome/data/Models/SpeechModel/Training/
# Save data to array file first

import torch as th
import torch.nn as nn

class KeywordAudioModel(nn.Module):
    '''
        This Network is designed to spot KeyWords in speech recognition.
        https://arxiv.org/pdf/2005.06720v2.pdf
        The structure of the network is designed to implement the following Paper.
    '''
    def __init__(self, input_shape=(1, 64, 72), lr=0.015, VERSION="0.01"):
        super(KeywordAudioModel, self).__init__()
        '''3x64x48'''
        self.batch_size = 32
        self._ConvolutionalLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5,7), padding=5, bias=True), # output (5 x 78 x 68)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4,4), stride=2), # Output (10 x 38 x 33)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(6,5), stride=4), # Output (20 x 9 x 8)
            nn.LeakyReLU(),
            nn.Flatten(),
        ) # Output 2600

        self._ReccurrentLayers = {
            'input_layer': nn.Linear(1440, 2200),
            # 'reccurrent_layer': nn.Linear(2800, 200)
        }

        # self._ReccurrentHidden = self._init_reccurrent()
        self.Data_Path = os.path.join(os.getcwd(), "data","Models","KeywordModel","Training")
        self._LinearLayers = nn.Sequential(
            nn.Linear(2200, 900, bias=True), 
            nn.LeakyReLU(),
            nn.Linear(900, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 2),
            nn.Softmax()
        )
        
        self._loss_func = nn.CrossEntropyLoss()
        self._optimizer = th.optim.Adam(self.parameters(), lr=lr)

        self.num_classes = 2
        self.VERSION = VERSION   
    # def _init_reccurrent(self):
    #     return th.zeros(shape=(1,200))  
    def forward(self, x):
        return self._LinearLayers(self._ReccurrentLayers['input_layer'](self._ConvolutionalLayers(x)))
    
    def preload_data(self):
        save_data_to_array(path=self.Data_Path)
        
    def load_in_data(self):
        data = []        
        for dir in os.listdir(os.path.join(self.Data_Path, "Mel_Imgs")):
            for file in os.listdir(os.path.join(self.Data_Path, "Mel_Imgs", dir)):
                data.append(
                    [np.array([np.load(os.path.join(self.Data_Path, "Mel_Imgs", dir, file), allow_pickle=True)]), int(dir)]
                )
                # print(data[-1][0].shape)
        data = np.array(data)
        
        return DataLoader(Dataset(data[:,0], data[:,1]), batch_size=self.batch_size, shuffle=True)
    
    
    def train(self, num_epochs, data_loader: DataLoader, save_location=None):
        from torch.autograd import Variable
        
        total_steps = len(data_loader)
          
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for mel_imgs, output in data_loader:
                
                self._optimizer.zero_grad()
                # mel_imgs = mel_imgs.to("cuda") if th.cuda.is_available() else mel_imgs
                mel_imgs = mel_imgs.float()
                output = self.forward(mel_imgs) 
                loss = self._loss_func(output.squeeze(), output)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                
                print(f"Epoch {epoch+1} Epoch Loss: {epoch_loss:.3f}")
        
        if save_location is None:
            save_location = os.path.join(os.getcwd(), "data","Models","KeywordModel","Version", self.VERSION)
        th.save(self.state_dict(), save_location)
                
                
                
                
                
                
                  
        