import os
import sys
import time as t

def main():
    # from src.utils import convert_path_to_wav
    # # convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')
    from src.Controllers import AudioController
    from src.utils import convert_path_to_wav
    # convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')
    # from src.Controllers import AudioController

    from src.Gwen.APIHandlers.SpotifyAPI.load_cookies import Spotify
    
    aud_cntrl = AudioController.AudioController()
    # aud_cntrl.set_stream_window(True)
    aud_cntrl.set_mode_sample()
    aud_cntrl.set_mode_collect_data(100, os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training',  'Sequences', 'Audio'), None)
    # from src.Gwen.AISystem.Networks import KeywordAudioModel
    # model = KeywordAudioModel(VERSION="0.02")
    # model.train(1000, model.load_in_data())

    while True:
        print('KeyWord Detection:', aud_cntrl.get_prediction()[0])
        t.sleep(0.25)
        

    
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
    y.play("Down on skidrow","muckraker")  # Adjust based on actual method names and parameters

    t.sleep(1000)

    y.quit()

if __name__ == '__main__':
    main()
