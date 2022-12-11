from enum import Enum

AudioControllerClass = None
CommandControllerClass = None
UserControllerClass = None



def import_classes():
    global AudioControllerClass, CommandControllerClass, UserControllerClass
    from ..Controllers.AudioController import AudioController
    from ..Controllers.CommandController import CommandController
    # from ..Controllers.UserController import UserController
    AudioControllerClass = AudioController
    CommandControllerClass = CommandController
    # UserControllerClass = UserController

class Gwen(Object):
    class States(Enum):
        SPOTIFY = 0
        CHROME = 1
        NETFLIX = 2
        TRAINING = 3
        USERS = 4
        STANDBY = 5

    def __init__(self,):
        self._current_state = self.States.STANDBY