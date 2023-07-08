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
    #     print('KeyWord Detection:', aud_cntrl.get_prediction())
    #     t.sleep(0.2)
        

    
    # aud_cntrl = AudioController.AudioController()
    # aud_cntrl.set_stream_window(True)
    # # aud_cntrl.set_mode_collect_data(10, os.path.join(os.getcwd(), 'data', 'Models', 'KeywordModel', 'Training', 'Mel_Imgs', '0'), None)
    # while True:
    #     t.sleep(1.0)
    # n = Netflix()
    # n.watch("bojack horseman")

    # t.sleep(100)

    # n.quit()
    
# if __name__ == '__main__':
#     main()

   # Assuming the class name is 'YouTube'


    # y = YouTube()
    # y.play("Down on skidrow","muckraker")  # Adjust based on actual method names and parameters

    # t.sleep(1000)

    # y.quit()

if __name__ == '__main__':
    main()
