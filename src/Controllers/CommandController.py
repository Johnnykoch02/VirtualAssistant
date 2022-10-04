

class CommandController(object):
    def __init__(self):
        
        self.commands = {
            'spotify': [
                'play', 'shuffle', 'search','playlist','volume'
            ],
            'led': [
                'mode', 'party', 
            ]
        }
