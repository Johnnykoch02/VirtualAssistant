import os
import sys
import numpy as np

import time as t

def main():
    # from src.utils import convert_path_to_wav
    # # convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')
    # # from src.Controllers import AudioController
    from src.utils import convert_path_to_wav
    # convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')
    from src.Controllers import AudioController
    
    aud_cntrl = AudioController.AudioController()
    # aud_cntrl.set_stream_window(True)
    # aud_cntrl.set_mode_sample()
    # aud_cntrl.set_mode_collect_data(100, os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Mel_Imgs', '1'), None)
    
    from src.Gwen.AISystem.Networks import KeywordAudioModel
    
    model = KeywordAudioModel.Load_Model(os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Version', '0.01'))
    # model.train(1000, model.load_in_data())
    
    while True:
        t.sleep(0.25)
        print(f" Prediction: {model.predict(np.array([aud_cntrl.buffer_to_img()]))}")

    
    # aud_cntrl = AudioController.AudioController()
    # aud_cntrl.set_stream_window(True)
    # # aud_cntrl.set_mode_collect_data(10, os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Mel_Imgs', '0'), None)
    # while True:
    #     t.sleep(1.0)

    # from src.ApplicationInterface.Netflix.interface import Netflix

    # n = Netflix()
    # n.watch("breaking bad")

    # t.sleep(10)

    # n.quit()


if __name__ == '__main__':
    main()

 