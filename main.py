import os
import sys

import time as t

def main():

    # from src.utils import convert_path_to_wav
    # from src.Gwen.APIHandlers.SpotifyAPI.load_cookies import Spotify
    # convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')
    # from src.Controllers import AudioController

    
    # aud_cntrl = AudioController.AudioController()
    # aud_cntrl.set_stream_window(True)
    # aud_cntrl.set_mode_sample()
    # aud_cntrl.set_mode_collect_data(10000, os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Mel_Imgs', '0'), None)
    
    # from src.Gwen.AISystem.Networks import KeywordAudioModel
    
    # model = KeywordAudioModel()
    # model.train(10, model.load_in_data())
    
    # aud_cntrl = AudioController.AudioController()
    # aud_cntrl.set_stream_window(True)
    # # aud_cntrl.set_mode_collect_data(10, os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Mel_Imgs', '0'), None)
    # while True:
    #     t.sleep(1.0)

    from src.ApplicationInterface.Netflix.interface import Netflix
    from src.ApplicationInterface.Youtube.YoutubePlayer import YouTube
    # n = Netflix()
    # n.watch("bojack horseman")

    # t.sleep(100)

    # n.quit()
    
# if __name__ == '__main__':
#     main()

   # Assuming the class name is 'YouTube'


    y = YouTube()
    y.play("Tony+Ferguson")  # Adjust based on actual method names and parameters

    t.sleep(10)

    y.quit()

if __name__ == '__main__':
    main()

    


    