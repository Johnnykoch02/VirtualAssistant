import os
import sys

import time as t

def main():
    from src.utils import convert_path_to_wav
    from src.Gwen.APIHandlers.SpotifyAPI.load_cookies import Spotify
    # convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')
    # from src.Controllers import AudioController
    
    # aud_cntrl = AudioController.AudioController()
    # aud_cntrl.set_stream_window(True)
    # aud_cntrl.set_mode_sample()
    # aud_cntrl.set_mode_collect_data(10000, os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Mel_Imgs', '0'), None)
    
    # from src.Gwen.AISystem.Networks import KeywordAudioModel
    
    # model = KeywordAudioModel()
    # model.train(10, model.load_in_data())
    
    while True:
        print('sleeping')
        t.sleep(1.0)

if __name__ == '__main__':
    main()

 