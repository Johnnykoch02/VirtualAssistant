import threading as th
import time as t


class Signal(object):
    
    def __init__(self, emitter, listener, target=None): 
        self.emitter = emitter
        self.single = type(listener) != type([])
        



