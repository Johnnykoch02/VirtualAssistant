#Imports
import os
# from src.Gwen.AISystem.preprocess import *
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
# Save data to array file first

import torch as th
import torch.nn as nn

class NonLSTMKeywordAudioModel(nn.Module):
    '''
        This Network is designed to spot Keywords in speech recognition.
        The structure of the network is designed to implement the following Paper.
    '''
    def __init__(self, input_shape=(1, 112, 40), lr=0.003, lstm_hidden_size=128, lstm_layers=3, VERSION="0.01"):
        super(NonLSTMKeywordAudioModel, self).__init__()
        '''3x64x48'''
        self.batch_size = 16
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self._ConvolutionalLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,7), stride=3, padding=5, bias=True), # output (5 x 78 x 68)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)), # Output (10 x 38 x 33)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=2), # Output (20 x 9 x 8)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        ) # Output 640
        # self.Data_Path = os.path.join(os.getcwd(), "data","Models","KeywordModel","Training")
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True, )
        self.fc = nn.Sequential(
            nn.LayerNorm(self.lstm_hidden_size),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.lstm_hidden_size, out_features=self.lstm_hidden_size,),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.lstm_hidden_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, out_features=2),
        )
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.num_classes = 2
        self.VERSION = VERSION   
    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        
    def forward(self, x, hidden=None): # TODO: Fix this and Implement Hidden State :3
        batch_size, num_timesteps, c, h, w = x.size()
        x = self._ConvolutionalLayers(x.view(batch_size*num_timesteps, c, h, w))
        x, _ = self.lstm(x.view(batch_size, num_timesteps, -1))
        x = self.fc(x)
        return x
    
    def predict(self, x):
        pred =self.forward(x).detach().cpu().float().squeeze(0).squeeze(0)
        probs = th.nn.functional.softmax(pred, dim=0).numpy()
        print(probs)
        return (pred[1] > 0.7 and pred.argmax(dim=0) == 1).item() # Test this 
        
    
    @staticmethod
    def Load_Model(model_path):
        model = NonLSTMKeywordAudioModel()
        model.load_state_dict(th.load(model_path))
        model.eval()
        return model
    
    def save_checkpoint(self, file=None):
        print('[KeywordModel] Saving Checkpoint...')
        if file != None:
            th.save(self.state_dict(), file)
        elif self.checkpoint_file != None:
            th.save(self.state_dict(), self.checkpoint_file)
    
                

class LSTMKeywordAudioModel(nn.Module):
    '''
        This Network is designed to spot KeyWords in speech recognition.1
        The structure of the network is designed to implement the following Paper.
    '''
    def __init__(self, input_shape=(1, 112, 40), lr=0.003, lstm_hidden_size=128, lstm_layers=3, VERSION="0.01"):
        super(LSTMKeywordAudioModel, self).__init__()
        '''3x64x48'''
        self._batch_size = 1
        self.input_shape = input_shape
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self._ConvolutionalLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,7), stride=3, padding=5, bias=True), # output (5 x 78 x 68)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)), # Output (10 x 38 x 33)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=2), # Output (20 x 9 x 8)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        ) # Output 640
        # self.Data_Path = os.path.join(os.getcwd(), "data","Models","KeywordModel","Training")
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True, )
        self.fc = nn.Sequential(
            nn.LayerNorm(self.lstm_hidden_size),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.lstm_hidden_size, out_features=self.lstm_hidden_size,),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.lstm_hidden_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, out_features=2),
        )
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self._hidden = self.init_hidden(self._batch_size)
        self.to(self.device)
        self.num_classes = 2
        self.VERSION = VERSION   
    
    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        
    def forward(self, x, hidden=None): # TODO: Fix this and Implement Hidden State :3
        batch_size, num_timesteps, c, h, w = x.size()
        x = self._ConvolutionalLayers(x.view(batch_size*num_timesteps, c, h, w))
        hidden = self.get_hidden()
        x, hidden = self.lstm(x.view(batch_size, num_timesteps, -1), (th.nan_to_num(hidden[0]), th.nan_to_num(hidden[1])))
        x = self.fc(x)
        self._hidden = hidden
        return x
    
    def predict(self, x):
        pred =self.forward(x).detach().cpu().float().squeeze(0).squeeze(0)
        probs = th.nn.functional.softmax(pred, dim=0).numpy()
        print(probs)
        return (pred[1] > 0.7 and pred.argmax(dim=0) == 1).item()
    
    def get_hidden(self):
        return self._hidden if self._hidden is not None else self.init_hidden()
    
    def init_hidden(self, batch_size):
        hidden_state = th.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(self.device)
        cell_state = th.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(self.device)
        # print(hidden_state.shape, cell_state.shape)
        return (hidden_state, cell_state)
    
    def reset_hidden_state(self, batch_size=None):
        if batch_size is None: batch_size = self._batch_size
        self._hidden = self.init_hidden(batch_size)
        # print(self._hidden.shape)
    
    @staticmethod
    def Load_Model(model_path):
        model = LSTMKeywordAudioModel()
        model.load_state_dict(th.load(model_path))
        model.eval()
        return model
    
    def save_checkpoint(self, file=None):
        print('[KeywordModel] Saving Checkpoint...')
        if file != None:
            th.save(self.state_dict(), file)
        elif self.checkpoint_file != None:
            th.save(self.state_dict(), self.checkpoint_file)                
                
                
# model = LSTMKeywordAudioModel()       
# print(model(th.zeros((1, 1, 1, 112, 40), device='cuda'), ).size())
