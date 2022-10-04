import speech_recognition as sr
import threading as th
from enum import Enum
from collections import deque
import time as t


class AudioController(object):
#'''This class is designed to listen into an Audio stream and listen for a Specific Keyword and deciper it'''
    class AudioStates(Enum):
        STANDBY = 0
        ENGAGED = 1
        NEW_USER = 2

    def __init__(self, keyword) -> None:
        self.States = AudioController.AudioStates
        self.mic = sr.Microphone()
        self.keyword = keyword
        self.state = self.States.STANDBY
        self.msg = deque()
        self.toggle = False
        self.new_user_Wait_flag = False
        self.audio_th = th.Thread(target=self.audio_processor)
        self.r = sr.Recognizer()
        self.audio_th.start()


    def audio_processor(self):
        while True:
            if self.state == self.States.STANDBY:
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
            elif self.state == self.States.NEW_USER:
                with self.mic as source:
                                self.wait_for_flag()
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
                    self.msg.append(tempStr)
                    self.toggle = True
                    self.state = self.States.STANDBY

                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                          print("Could not request results; {0}".format(e))

    def wait_for_flag(self):
        if (self.new_user_Wait_flag):
            return
        else:
            t.sleep(0.05)
            self.wait_for_flag()
    
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

gwen = AudioController("Hey Gwen")