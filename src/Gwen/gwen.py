from enum import Enum

import os
import time as t
import threading as thr
import re
from collections import deque

AudioControllerClass = None
CommandControllerClass = None
UserControllerClass = None

SpotifyClass = None
NetflixClass = None
YouTubeClass = None

def get_gwen_instance():
    return None

class Gwen:
    GwenInstance = None
    # ----- Gwen Contexts -----
    # TODO: ADD context threading for execution to allow for gwen to run in different threads.
    class Context:
        def __init__(self, obj, data):
            self.obj = obj # Maybe needs some sort of Target Data ??? Well see
            self.data = data
            self._exec_thr = None
            self._run_thr = None
            
        def exec(self, kwargs):
            # Demo of how to Execute a Context
            self._exec_thr = thr.Thread(target=self.obj.exec, args=(self.data), daemon=True)
            self._exec_thr.start()
            self._exec_thr.join()
            return self.obj.exec(self.data)
        
        def run(self, is_main_context =False) -> None:
            self.obj.run(self.data, is_main_context) # TODO: Actually implement this
        
        def quit(self,) -> None:
            self.obj.quit() # TODO: Actually implement this

        def validate_exec(self, kwargs) -> bool:
            pass
        
        def __repr__(self) -> str:
            return str(f'{self.obj}: {self.data}')
        
        def __str__(self) -> str:
            return self.__repr__()
    
    class PassiveContext(Context):
        def __init__(self, data): # Fix this for Cmd Parser
            super(Gwen.PassiveContext, self).__init__(Gwen.Gwen()._AudioController, '')
            
        def run(self, is_main_context=False) -> None: # Run Audio Controller
            return super().run(is_main_context)
        
        def quit(self,) -> None:
            self.obj.quit()
        
    class SpotifyContext(Context):
        def __init__(self, data):
            super(Gwen.SpotifyContext, self).__init__(SpotifyClass(), data)
            
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context)    
        
        def quit(self,) -> None:
            self.obj.stop()

        def exec(self, kwargs):
            if kwargs['func'] == "play":
                self.obj.search((kwargs['song']) + " " +kwargs.get('artist', "").strip())
            elif kwargs['func'] == "pause":
                self.obj.pause()
        
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['func'] == "play" and kwargs.get('song', None) != None and kwargs.get('artist', "") != None or kwargs ['func'] == "pause")
        
    class YouTubeContext(Context):
        def __init__(self, data):
            super(Gwen.YouTubeContext, self).__init__(YouTubeClass(), data)
            
        def run(self, is_main_context=False) -> None:
            return super().run(is_main_context) 
        
        def quit(self,) -> None:
            self.obj.quit()
            
        def exec(self, kwargs):
            self.obj.play(kwargs['query'], channel_name = kwargs.get('channel_name', None))   
        
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['func'] == "play" and kwargs.get('query', None) != None)
        
    class NetflixContext(Context):
        def __init__(self, data):
            super(Gwen.NetflixContext, self).__init__(NetflixClass(), data)  
             
        def run(self, is_main_context=False) -> None:
            return super().run(self.data, is_main_context)
        
        def quit(self,) -> None:
            self.obj.quit()
        
        def exec(self, kwargs):
            if kwargs['func'] == "play":
                self.obj.play(kwargs['query'])
            
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['func'] == "watch" and kwargs.get('query', None) != None)
             
    class GwenContext(Context):
        def __init__(self, data):
            super(Gwen.GwenContext, self).__init__(Gwen.Gwen(), data)
        
        def run(self, is_main_context=False) -> None:
            if not 'clear_context' in self.data['target']:
                super().run(is_main_context)

        def quit(self,) -> None:
            # self.obj.quit()
            pass
        
        def exec(self, kwargs):
            if kwargs['func'] == "output_speech":
                pass
            elif kwargs['func'] == "clear_context":
                self.obj.clear_context()
            elif kwargs['func'] == "collect_keyword_data":
                self.obj.collect_keyword_data(kwargs['num_samples'])
        
        def validate_exec(self, kwargs) -> bool:
            assert isinstance(kwargs, dict) 
            return (kwargs['func'] == "output_speech" and kwargs.get('text', None) != None or kwargs ['func'] == "clear_context" or kwargs ['func'] == "collect_keyword_data" and kwargs.get('num_samples', None) != None and isinstance(kwargs['num_samples'], int))

    # --- Gwen Methods --- 
    def __new__(cls, *args, **kwargs):
            if not cls.GwenInstance:
                cls.GwenInstance = super(Gwen, cls).__new__(cls, *args, **kwargs)
            return cls.GwenInstance
        
    def __init__(self,):
        '''
        Initializes Gwen Instance,
        '''
        print('Initializing Gwen Virtual Assistant...')
        def import_classes():
            global AudioControllerClass, CommandControllerClass, UserControllerClass, SpotifyClass, NetflixClass, YouTubeClass
            from ..Controllers.AudioController import AudioController
            from ..Controllers.CommandController import CommandController
            from src.Gwen.APIHandlers.SpotifyAPI.load_cookies import Spotify
            from src.ApplicationInterface.Netflix.interface import Netflix
            from src.ApplicationInterface.Youtube.YoutubePlayer import YouTube
            AudioControllerClass = AudioController
            CommandControllerClass = CommandController
            SpotifyClass = Spotify
            YouTubeClass = YouTube
            NetflixClass = Netflix
            # TODO: Init the Context Object Classes
            
        import_classes()
        self._AudioController = AudioControllerClass(self,)
        self._CommandController = CommandControllerClass(self,)
        self._current_context = self.PassiveContext("")
        self._Contexts = deque([self._current_context], maxlen = 25)
        
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
        self._CommandController.ProcessCommand(cmd, self._current_context)
        
    def add_context(self, context) -> None:
        self._current_context = context
        self._Contexts.appendleft(context)
    
    def run_context(self,) -> None:
        current_context = self._current_context
        [context.run(is_main_context = current_context == context) for context in list(self._Contexts)]
    
    def end_context(self, context) -> None:
        context.quit()
        if context == self._current_context:
            self._current_context = self._Contexts.popleft()
        else:
            self._Contexts.remove(context)
        
    def clear_context(self,) -> None:
        """
        Clears the Current Context of Gwen.
        """
        [context.quit() for context in list(self._Contexts)]
        self._current_context = self.PassiveContext("")
        self._Contexts = deque([self._current_context])
        
    def speech_output(self, text):
        """
        Uses the Configured TTS Engine to speak the given text.
        """
        pass # TODO: Actually implement this
        