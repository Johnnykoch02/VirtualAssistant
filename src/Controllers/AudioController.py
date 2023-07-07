import speech_recognition as sr
import pyaudio
from pydub import AudioSegment
import threading as th
from enum import IntEnum
from collections import deque
import numpy as np
import os
import time as t
import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')
from src.Gwen.AISystem.Networks import KeywordAudioModel
import torch as torch
from src.utils import get_mel_image_from_float_normalized, normalize_mfcc, get_json_variables

class AudioController(object):
    class State(object):
        class Mode(IntEnum):
            LISTENING = 0
            STREAMING = 1
            TEMP = 2
            
            DATA_COLLETION = 5
            TRAP = 10
            
        def __init__(self, ):
            self.mode = AudioController.State.Mode.LISTENING
        
        def transition(self, set=-1):
            if not set == -1:
                self.mode = AudioController.State.Mode(set)
                return self.mode != AudioController.State.Mode.TRAP
            self.mode = AudioController.State.Mode((int(self.mode) + 1) % 3)
            
        def __call__(self,):
            return self.mode
    
    def __init__(self, GwenInstance):
        '''Basic Audio Control'''
        self.GwenInstance = GwenInstance
        self.States = AudioController.AudioStates
        self.mic = pyaudio.PyAudio()
        self._audio_th = th.Thread(target=self.audio_processor)
        self.img_th = th.Thread(target= self.AudioStreamWindow)
        self.r = sr.Recognizer()
        self._audio_buffer = deque() 
        
        self.state = AudioController.State()
        
        self._config = get_json_variables(os.path.join(os.getcwd(), 'data', 'Gwen', 'Audio', 'AudioControllerConfig.json'), ["model_path","data_output_path","n_mfcc","max_len"])
        
        self.n_mfcc = self._config["n_mfcc"]
        self.max_len = self._config["max_len"]
        
        self.__audio_buffer_lock = th.Lock()
        with self.__audio_buffer_lock :
            for _ in range(4):
                self._audio_buffer.append(np.zeros(shape=(self.n_mfcc, self.max_len), dtype=np.float32)) 
                
        self._current_stream_img = self.buffer_to_img()    
        self._prediction_model = KeywordAudioModel.Load_Model(os.path.join(os.getcwd(), self._config["model_path"]))
        self._data_output_path = os.path.join(os.getcwd(), self._config["data_output_path"])
        
        
        self._audio_th.start()
        self._stream = self.record_audio()
        
        
    def buffer_to_img(self):
        with self.__audio_buffer_lock:
           return normalize_mfcc(np.vstack(np.array(self._audio_buffer).copy()))
       
    def record_audio(self) -> pyaudio.Stream:
        """
        Streams Microphone Audio and stores it in a buffer as a Mel Image.
        """
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
    
    # --- Application API --- #

    def run(self, is_main_context=False) -> None:
        if self.state() == AudioController.State.Mode.LISTENING:
            # Run Prediction on the Current Audio Stream 
            prediction = self.get_prediction()
            if prediction: # Stop the stream and Transition to Command Parsing
                self._stream.stop_stream()
                self.state.transition()
                
        elif self.state() == AudioController.State.Mode.STREAMING:
            # Use Microphone Audio to Predict Command-String
            with self.mic as source:
                    self.r.adjust_for_ambient_noise(source=source, duration=0.2)                                                             
                    try:
                        temp_Audio = self.r.listen(source) 
                        response = openai.Audio.transcribe("whisper-1",temp_Audio.get_wav_data())
                        self.state.transition()
                        # self.GwenInstance. Do something with response["text"]
                    except sr.UnknownValueError as e:
                        print("Could not understand audio")
                        self.state.transition(AudioController.State.Mode.LISTENING)
                    except sr.RequestError as e:
                        print("Could not request results; {0}".format(e))
                        self.state.transition(AudioController.State.Mode.LISTENING)
                        
        elif self.state() == AudioController.State.Mode.TEMP:
            pass
        
        elif self.state() == AudioController.State.Mode.DATA_COLLETION:
            self.state.transition(0) # Asssuming Context took care of the rest.
        
        elif self.state() == AudioController.State.Mode.TRAP:
            '''Bad state :( hope I don't end up here '''
        
    def transition_mode(self, mode=-1):
        self.state.transition(mode)
            
    def get_prediction(self,):
        img = self.buffer_to_img()
        while len(img.shape) < 5:
            img = np.expand_dims(img, axis=0)
        try:
            return self._prediction_model.predict(torch.tensor(img, device=self._prediction_model.device))
        except:
            return False # TODO: Add Logging of errors.
        
        
    def exec(self, data):
        if data['target'] == 'DataCollection':
            self.state.transition(AudioController.State.Mode.DATA_COLLETION)
            self.collect_data_sequence(num_samples=data['num_samples'], path=self._data_output_path)
        elif data['target'] == '':
            pass # Add more Cases eventually.
        
    # --- Data Collection API --- #
    def collect_data_sequence(self, num_samples, path, sample_episode=100, time_low=5, time_high=20):
        print('While collecting data, you will be prompted to provide audio. When prompted, make sure to respond promptly.\nThis will ensure proper data collection.')
        for _ in range(num_samples):
            with self.mic:  
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
            
        self.state.transition(AudioController.State.Mode.LISTENING) # Transition to listening mode
        

        
            
        
            
    