import speech_recognition as sr
import pyaudio
from pydub import AudioSegment
import threading as th
from enum import Enum
from collections import deque
import numpy as np
import os
import time as t
from src.Gwen.AISystem.Networks import KeywordAudioModel
import torch as torch
from src.utils import get_mel_image_from_float_normalized, normalize_mfcc

# import cv2


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
        self.mic = pyaudio.PyAudio()
        self.audio_th = th.Thread(target=self.audio_processor)
        self.img_th = th.Thread(target= self.AudioStreamWindow)
        self.r = sr.Recognizer()
        
        self.keyword = keyword
        self.state = self.States.SAMPLING
        self._audio_buffer = deque() 
        self.n_mfcc = 28
        self.max_len = 40
        self.__audio_buffer_lock = th.Lock()
        with self.__audio_buffer_lock :
            for _ in range(4):
                self._audio_buffer.append(np.zeros(shape=(self.n_mfcc, self.max_len), dtype=np.float32))
        self._current_stream_img = self.buffer_to_img()
        self._prediction_model = KeywordAudioModel.Load_Model(os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Checkpoints', 'KeywordCheckpoint_1000.zip'))
        
        '''New User Stuff'''
        self.toggle = False
        self.new_user_Wait_flag = False
        
        '''Data Collection Stuff'''
        self._data_path = os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Sequences', 'Audio')
        self._step_number = 0
        self._stream_window_visble = stream_visible
        self._sample_episode = 100
        
        self.audio_th.start()
        self.audio_stream = self.record_audio()
    
    def record_audio(self):
        audio_format = pyaudio.paInt16
        num_channels = 1
        sample_rate = 48000
        duration = 0.25        
        frame_length = int(sample_rate*duration)

        def __audio_stream_callback__(in_data, a, b, c):
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_data_np = np.frombuffer(audio_data.tobytes(), dtype=np.int16).astype(np.float32)
            audio_data_np /= np.iinfo(np.int16).max # Norm the stream data
            with self.__audio_buffer_lock: # Using this buffer lock ensures that we do not read from a different thread
                self._audio_buffer.popleft()
                mel_img = get_mel_image_from_float_normalized(audio_data_np, sound_rate=sample_rate)
                self._audio_buffer.append(mel_img)
            return (in_data, pyaudio.paContinue)

        # Open stream on our microphone
        stream = self.mic.open(format=audio_format,channels=num_channels, rate=sample_rate, input=True, frames_per_buffer=frame_length, stream_callback=__audio_stream_callback__)
        stream.start_stream()
        return stream
        
    def set_mode_collect_data(self, num_steps, path, sample_episode=None):
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
        return self.buffer_to_img()
    
    def collect_data(self, num_samples, path, sequence=False, sample_episode=100):
        if sequence:
            self.collect_data_sequence(num_samples, path, sample_episode=sample_episode)
        else:
            self.collect_data_simple(num_samples, path, sample_episode=sample_episode)
    
    def collect_data_sequence(self, num_samples, path, sample_episode=100, time_low=2, time_high=20):
        print('While collecting data, you will be prompted to provide audio. When prompted, make sure to respond promptly.\nThis will ensure proper data collection.')
        continuous = num_samples == -1
        
        if continuous:
            num_samples = sample_episode
            
        if self._sample_episode != 100:
            self._sample_episode = 100
        for _ in range(num_samples):
            with self.mic as source:  
                index = len(os.listdir(path))
                record_time = np.random.randint(low=time_low, high=time_high)
                try:  
                    print('Recording for ' + str(record_time) + ' seconds...')  
                    t.sleep(1.0)
                    print('--- START ---')
                    print('...Collecting...')
                        
                    audio = self.r.record(source=self.mic, duration=record_time)
                    
                    print('--- Finished Recording Sample ---')
                    with open(os.path.join(path, f'{index}.wav'), 'wb') as f:
                        f.write(audio.get_wav_data())
                    print('--- Finished Saving Sample (\..Enter to Continue../) ---')
                    input()
                        # t.sleep(1.0)
                except sr.UnknownValueError:
                        print("Could not understand audio")
                except sr.RequestError as e:
                        print("Could not request results; {0}".format(e))
            
        if continuous:
            if input("Enter \'n\' to break...") == 'n':
                self.state = self.States.SAMPLING
                return
            self.collect_data_sequence(-1, path, sample_episode)
            
        self.set_mode_sample()
        
    def collect_data_simple(self, num_samples, path, sample_episode=100):
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
                            temp_buffer.append(self.r.record(source=self.mic, duration=0.25).frame_data)
                        
                        print('--- Finished Sample ---')
                        audio_img = np.concatenate(list(temp_buffer), axis=1)
                        if self._stream_window_visble:
                            self._current_stream_img = audio_img
                            
                        np.save(os.path.join(path, f'{index}.npy'), arr=audio_img, allow_pickle=True)
                        # t.sleep(1.0)
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
    def get_prediction(self,):
        img = self.buffer_to_img()
        while len(img.shape) < 5:
            img = np.expand_dims(img, axis=0)
        # print(img, img.shape)
        try:
            return self._prediction_model.predict(torch.tensor(img, device=self._prediction_model.device))
        except:
            return None
    def audio_processor(self):
        
        while True:
            t.sleep(0.1)
            # self.state = self.Gwen.get_state()
            if self.state == self.States.SAMPLING:
                pass
            #     # print("Hello")
            #     with self.mic as source:     
            #         self._audio_buffer.popleft()     
            #         temp_Audio = self.r.record(source=self.mic, duration=0.25)
            #         try:
            #             audio_bytes = temp_Audio.frame_data

            #             # Convert raw bytes to an AudioSegment
            #             audio_segment = AudioSegment.from_raw(audio_bytes, sample_width=temp_Audio.sample_width,
            #                           frame_rate=temp_Audio.sample_rate, channels=temp_Audio.channels)
            #             mel_img = get_mel_image_from_float_normalized(audio_segment, sound_rate=temp_Audio.sample_rate)
            #             self._audio_buffer.append(mel_img)
                        
            #             # if self._stream_window_visble:
            #             #     self._current_stream_img = self.buffer_to_img()
                            
            #         except sr.UnknownValueError:
            #             print("Could not understand audio")
            #         except sr.RequestError as e:
            #               print("Could not request results; {0}".format(e))
                          
            elif self.state == self.States.DATA_COLLECTION:             
                self.collect_data(self._step_number, self._data_path, True, self._sample_episode)
            
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
        with self.__audio_buffer_lock:
           return normalize_mfcc(np.vstack(np.array(self._audio_buffer).copy()))
   
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
                pass
                # cv2.imshow('Audio Stream', self._current_stream_img)
                # cv2.waitKey(500)
                # cv2.destroyAllWindows()
            t.sleep(0.25)