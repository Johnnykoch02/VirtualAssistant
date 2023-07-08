import os
import sys
import time as t

def main():
    # from src.utils import convert_path_to_wav
    # # convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')
    from src.Controllers import AudioController

    from src.Gwen.gwen import Gwen
    # aud_cntrl = AudioController.AudioController(None)

    while True:
        Gwen.Gwen().run_context()
        t.sleep(0.08) # 
    #     print('KeyWord Detection:', aud_cntrl.get_prediction())
    #     t.sleep(0.2)

if __name__ == '__main__':
    main()
