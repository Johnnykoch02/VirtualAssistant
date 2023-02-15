#Imports
import os
from src.Gwen.AISystem.preprocess import *
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader
from keras.preprocessing.image import ImageDataGenerator

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
    def __init__(self, input_shape=(1, 64, 72), lr=0.08, VERSION="0.01"):
        super(KeywordAudioModel, self).__init__()
        '''3x64x48'''
        self.batch_size = 32
        self._ConvolutionalLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5,7), padding=5, bias=True), # output (5 x 78 x 68)
            nn.BatchNorm2d(5),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4,4)), # Output (10 x 38 x 33)
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(2,2), stride=2), # Output (20 x 9 x 8)
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        ) # Output 1440

        # self._ReccurrentLayers = {
        #     'input_layer': nn.Linear(1440, 2200),
        #     # 'reccurrent_layer': nn.Linear(2800, 200)
        # }

        # self._ReccurrentHidden = self._init_reccurrent()
        self.Data_Path = os.path.join(os.getcwd(), "data","Models","KeywordModel","Training")
        self._LinearLayers = nn.Sequential(
            nn.Linear(1440, 900, bias=True), 
            nn.LeakyReLU(),
            nn.Linear(900, 2),
            nn.Softmax(dim=1)
        )
        
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.to(self.device)
        
        self._loss_func = nn.CrossEntropyLoss()
        self._optimizer = th.optim.RMSprop(self.parameters(), lr=lr)

        self.num_classes = 2
        self.VERSION = VERSION   
        
        self._training_data = self.load_in_data(fit=True)
        self.normalizer = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True) #Issues with normalization: Need to convert for Pytorch compatability
        self.normalizer.fit(self._training_data)

        
    # def _init_reccurrent(self):
    #     return th.zeros(shape=(1,200))  
    def forward(self, x):
        x = x.to(self.device)
        return self._LinearLayers(self._ConvolutionalLayers(x))
    
    def preload_data(self):
        save_data_to_array(path=self.Data_Path)
        
    def load_in_data(self, fit=False):
        data = []        
        if not fit:
            for dir in os.listdir(os.path.join(self.Data_Path, "Mel_Imgs")):
                for i, file in enumerate(os.listdir(os.path.join(self.Data_Path, "Mel_Imgs", dir))):
                    if i > 1000: # Temp Fix for large num of files, get rid of later
                        break
                    img = np.load(os.path.join(self.Data_Path, "Mel_Imgs", dir, file), allow_pickle=True)
                    data.append(
                        [np.array(self.normalizer.standardize([img])), int(dir)]
                    )
            data = np.array(data)
            return DataLoader(Dataset(data[:,0], data[:,1]), batch_size=self.batch_size, shuffle=True)
        else:
            for dir in os.listdir(os.path.join(self.Data_Path, "Mel_Imgs")):
                for file in os.listdir(os.path.join(self.Data_Path, "Mel_Imgs", dir)):
                    img = np.load(os.path.join(self.Data_Path, "Mel_Imgs", dir, file), allow_pickle=True)
                    data.append(
                        [img]
                    )
            return np.array(data)
    
    def predict(self, x):
        x = th.from_numpy(self.normalizer.standardize(x)).to(self.device)
        return th.argmax(self.forward(x)).item()
        
    
    def train(self, num_epochs, data_loader: DataLoader, save_location=None):
        from torch.autograd import Variable
        
        total_steps = len(data_loader)
          
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for mel_imgs, output in data_loader:
                
                self._optimizer.zero_grad()
                # mel_imgs = mel_imgs.to("cuda") if th.cuda.is_available() else mel_imgs
                mel_imgs = mel_imgs.float()
                out = self.forward(mel_imgs) 
                output = th.nn.functional.one_hot(output, self.num_classes).float().to(self.device)

                loss = self._loss_func(out, output)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                
                print(f"Epoch {epoch+1} Epoch Loss: {epoch_loss:.3f}")
        
        if save_location is None:
            save_location = os.path.join(os.getcwd(), "data","Models","KeywordModel","Version", self.VERSION)
        th.save(self.state_dict(), save_location)
                
                
                
                
                
                
                  
        