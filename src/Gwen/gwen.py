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


class Gwen:
    
    GwenInstance = None
    # ----- Gwen Contexts -----
    class Context:
        def __init__(self, obj, data):
            self.obj = obj # Maybe needs some sort of Target Data ??? Well see
            self.data = data
            
        def exec(self,):
            return self.obj.exec(self.data) # TODO: Actually implement this
        
        def run(self, is_main_context =False) -> None:
            self.obj.run(self.data, is_main_context) # TODO: Actually implement this
        
        def __repr__(self):
            return str(f'{self.obj}: {self.data}')
    
    class PassiveContext(Context):
        def __init__(self,):
            super(Gwen.PassiveContext, self).__init__(None, '')
            
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context)
        
    class SpotifyContext(Context):
        def __init__(self, data):
            super(Gwen.SpotifyContext, self).__init__(SpotifyClass(), data)
            
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context)    
        
    class YoutubeContext(Context):
        def __init__(self, data):
            super(Gwen.YoutubeContext, self).__init__(YouTubeClass(), data)
            
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context)    
        
    class NetflixContext(Context):
        def __init__(self, data):
            super(Gwen.NetflixContext, self).__init__(NetflixClass(), data)   
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context)
                 
    class SpeechContext(Context):
        def __init__(self, data):
            super(Gwen.YoutubeContext, self).__init__(SpeechClass(), data)


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
    
    def set_context(self, context):
        pass
    
    def run_context(self, context):
        [self._current_context.run(is_main_context = self._current_context == context)for context in self._Contexts] #TODO: Actually implement this
    
    def clear_context(self,):
        """
        Clears the Current Context of Gwen.
        """
        self._current_context = self.PassiveContext()
        