from enum import Enum

import os
import time as t
import re
from collections import deque

AudioControllerClass = None
CommandControllerClass = None
UserControllerClass = None

SpotifyClass = None
NetflixClass = None
YouTubeClass = None
SpeechClass = None

def get_gwen_instance():
    return None

class Gwen:
    
    GwenInstance = None
    # ----- Gwen Contexts -----
    class Context:
        def __init__(self, obj, data):
            self.obj = obj # Maybe needs some sort of Target Data ??? Well see
            self.data = data
            
        def exec(self, kwargs):
            return self.obj.exec(self.data) # TODO: Actually implement this
        
        def run(self, is_main_context =False) -> None:
            self.obj.run(self.data, is_main_context) # TODO: Actually implement this

        def validate_exec(self, kwargs) -> bool:
            pass
        
        def __repr__(self) -> str:
            return str(f'{self.obj}: {self.data}')
    
    class PassiveContext(Context):
        def __init__(self, data): # Fix this for Cmd Parser
            super(Gwen.PassiveContext, self).__init__(Gwen.Gwen()._AudioController, '')
            
        def run(self, is_main_context=False) -> None: # Run Audio Controller
            return super().run(is_main_context)
        
    class SpotifyContext(Context):
        def __init__(self, data):
            super(Gwen.SpotifyContext, self).__init__(SpotifyClass(), data)
            
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context)    

        def exec(self, kwargs):
            pass
        
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['target'] == "play" and kwargs.get('song', None) != None and kwargs.get('artist', None) != None or kwargs ['target'] == "pause")
        
    class YoutubeContext(Context):
        def __init__(self, data):
            super(Gwen.YoutubeContext, self).__init__(YouTubeClass(), data)
            
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context) 
        
        def exec(self, kwargs):
            pass   
        
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['target'] == "play" and kwargs.get('query', None) != None)
        
    class NetflixContext(Context):
        def __init__(self, data):
            super(Gwen.NetflixContext, self).__init__(NetflixClass(), data)  
             
        def run(self, is_main_context=False) -> None:
            return super().run(self.data, is_main_context)
        
        def exec(self, kwargs):
            if kwargs['target'] == "play":
                self.obj.play(kwargs['query'])
            
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['target'] == "watch" and kwargs.get('query', None) != None)
             
    class GwenContext(Context):
        def __init__(self, data):
            super(Gwen.GwenContext, self).__init__(Gwen.Gwen(), data)
        
        def run(self, is_main_context=False) -> None:
            super().run(is_main_context)
        
        def exec(self, kwargs):
            pass
        
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['target'] == "output_speech" and kwargs.get('text', None) != None or kwargs ['target'] == "clear_context" or kwargs ['target'] == "collect_keyword_data" and kwargs.get('num_samples', None) != None and isinstance(kwargs['num_samples'], int))

    # --- Gwen Methods --- 

    def __init__(self,):
        '''
        Initializes Gwen Instance,
        '''
        def import_classes():
            global AudioControllerClass, CommandControllerClass, UserControllerClass
            from ..Controllers.AudioController import AudioController
            from ..Controllers.CommandController import CommandController
            # from ..Controllers.UserController import UserController
            AudioControllerClass = AudioController
            CommandControllerClass = CommandController 
            # TODO: Init the Context Object Classes
            
            
        import_classes()
        self._current_context = self.PassiveContext()
        self._AudioController = AudioControllerClass()
        self._CommandController = CommandControllerClass()
        
        self._Contexts = deque()
        
        
    @staticmethod
    def Gwen():
        if Gwen.GwenInstance is None:
            Gwen.GwenInstance = Gwen()            
        return Gwen.GwenInstance
    
    # ---- Gwen API --- 
    
    def execute_command(self, cmd):
        """
        Passes the Command to the Command Controller for Execution.
        """
        self._CommandController.ProcessCommand(cmd)
        
    def add_context(self, context) -> None:
        self._current_context = context
        self._Contexts.appendleft(context)
    
    def run_context(self,) -> None:
        [self._current_context.run(is_main_context = self._current_context == context) for context in self._Contexts] #TODO: Actually implement this
    
    def end_context(self, context) -> None:
        if context == self._current_context:
            self._current_context = self._Contexts.popleft()
        else:
            self._Contexts.remove(context)
        
    def clear_context(self,) -> None:
        """
        Clears the Current Context of Gwen.
        """
        self._current_context = self.PassiveContext()
        self._Contexts = deque([self._current_context])
        
    def speech_output(self, text):
        """
        Uses the Configured TTS Engine to speak the given text.
        """
        pass # TODO: Actually implement this
        