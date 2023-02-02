import speech_recognition as sr
import threading as th
from enum import Enum
from collections import deque
import numpy as np
import os
import time as t
import cv2


class AudioController(object):
#'''This class is designed to listen into an Audio stream and listen for a Specific Keyword and deciper it'''
    class AudioStates(Enum):
        SAMPLING = 0
        ENGAGED = 1
        PROCESSING = 2
        NEW_USER = 3
        DATA_COLLECTION = 4

    def __init__(self, keyword='', stream_visible=True):
        '''Basic Audio Control'''
        self.States = AudioController.AudioStates
        self.mic = sr.Microphone()
        self.audio_th = th.Thread(target=self.audio_processor)
        self.img_th = th.Thread(target= self.AudioStreamWindow)
        self.r = sr.Recognizer()
        
        self.keyword = keyword
        self.state = self.States.SAMPLING
        self._audio_buffer = deque()
        for _ in range(16):
            self._audio_buffer.append(np.zeros(shape=(72,4)))
        self._current_stream_img = self.buffer_to_img()
        
        '''New User Stuff'''
        self.toggle = False
        self.new_user_Wait_flag = False
        
        '''Data Collection Stuff'''
        self._data_path = None
        self._step_number = 0
        self._stream_window_visble = stream_visible
        self._sample_episode = 100
        
        self.audio_th.start()
    
    def set_mode_collect_data (self, num_steps, path, sample_episode=None):
        self._data_path = path
        self._step_number = num_steps
        if sample_episode is not None:
            self._sample_episode = sample_episode    
        self.state = self.States.DATA_COLLECTION
    
    def set_mode_sample(self):
        self.state = self.States.SAMPLING

    def set_mode_engaged(self):
        self.state = self.States.ENGAGED
    
    def set_mode_new_user(self):
        self.state = self.States.NEW_USER
    
    def get_audio_state(self):
        return np.expand_dims(self.buffer_to_img(), axis=0)
           
    def collect_data(self, num_samples, path, sample_episode=100):
        from ..Gwen.AISystem.preprocess import audioDataTomfcc
        print('While collecting data, you will be prompted to provide audio. When prompted, make sure to respond promptly.\nThis will ensure proper data collection.')
        continuous = num_samples == -1
        
        if continuous:
            num_samples = sample_episode
            
        if self._sample_episode != 100:
            self._sample_episode = 100
        for _ in range(num_samples):
            temp_buffer = deque()
            with self.mic as source:  
                    index = len(os.listdir(path))
                    try:
                        print('--- START ---')
                        print('...Collecting...')
                        for _ in range(16):  
                            temp_buffer.append(self.r.record(source=self.mic, duration=0.125))
                        temp_buffer = map(audioDataTomfcc, temp_buffer)
                        print('--- Finished Sample ---')
                        audio_img = np.concatenate(list(temp_buffer), axis=1)
                        if self._stream_window_visble:
                            self._current_stream_img = audio_img
                            
                        np.save(os.path.join(path, f'{index}.npy'), arr=audio_img, allow_pickle=True)
                        t.sleep(1.0)
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                          print("Could not request results; {0}".format(e))
            
        if continuous:
            if input("Enter \'n\' to break...") == 'n':
                self.state = self.States.SAMPLING
                return
            self.collect_data(-1, path, sample_episode)
            
        self.set_mode_sample()
  
    def audio_processor(self):
        from ..Gwen.AISystem.preprocess import audioDataTomfcc
        
        while True:
            # self.state = self.Gwen.get_s tate()
            if self.state == self.States.SAMPLING:
                # print("Hello")
                with self.mic as source:     
                    self._audio_buffer.popleft()     
                    temp_Audio = self.r.record(source=self.mic, duration=0.125)
                    try:
                        mel_img = audioDataTomfcc(temp_Audio)
                        self._audio_buffer.append(mel_img)
                        
                        # if self._stream_window_visble:
                        #     self._current_stream_img = self.buffer_to_img()
                            
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                          print("Could not request results; {0}".format(e))
                          
            elif self.state == self.States.DATA_COLLECTION:             
                self.collect_data(self._step_number, self._data_path, self._sample_episode)
            
            elif self.state == self.States.ENGAGED:
                with self.mic as source:
                    self.r.adjust_for_ambient_noise(source=source, duration=0.2)                                                             
                    temp_Audio = self.r.listen(source) 
                    temp_Msg = self.r.recognize_google(temp_Audio) 
                    print(temp_Msg)
                    try:
                        if temp_Msg == self.keyword:
                            print('Working')
                            self.state = self.States.ENGAGED
                    except sr.UnknownValueError:
                               print("Could not understand audio")
                    except sr.RequestError as e:
                                     print("Could not request results; {0}".format(e))

            else:
                with self.mic as source:                                                                       
                      temp_Audio = self.r.listen(source)  
                try:
                    tempStr = self.r.recognize_google(temp_Audio)
                    # self.msg.append(tempStr)
                    self.toggle = True
                    self.state = self.States.SAMPLING

                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                          print("Could not request results; {0}".format(e))

    # def wait_for_flag(self):
    #     if (self.new_user_Wait_flag):
    #         return
    #     else:
    #         t.sleep(0.05)
    #         self.wait_for_flag()
    
    def get_raw_audio_request(self):
        pass

    def read(self):
        if len(self.msg) > 0:
            return self.msg[0]
        else:    
            return None
    
    def reset(self):
        if len(self.msg) < 2:
            self.toggle = False
        self.msg.popleft()    

    def toggled(self):
        return self.toggle

    def AudioDataToNpy(self, sound):
        byte_data, sr, sw = sound.get_wav_data(), sound.sample_rate, sound.sample_width
        data_s16 = np.frombuffer(byte_data, dtype=np.int16, count=len(byte_data)//2, offset=0)
        float_data = data_s16 * 0.5**15 
        wave = float_data[::3]
        return wave, sr

    def buffer_to_img(self):
       return np.concatenate(list(self._audio_buffer), axis=1)
   
    # def set_stream_window(self, val:bool):
    #     self._stream_window_visble = val
    #     cv2.namedWindow('Audio Stream', cv2.WINDOW_NORMAL) if val else cv2.destroyAllWindows()
        
        # if val:
        #     self.img_th.start()
        # else:
        #     self.img_th.join()
    
    def AudioStreamWindow(self):
        while True:
            if self._stream_window_visble:
                cv2.imshow('Audio Stream', self._current_stream_img)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
            t.sleep(0.25)