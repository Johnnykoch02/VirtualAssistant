#Imports
import os
import subprocess
try:
    from main import PROJECT_PATH
except:
    pass
import json
import sys


###

def convert_path_to_JSON(path, var_name):
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        json_file.close()
    return data[var_name]
# path= os.path.abspath()
# print(os.path.join(path,"configurationState.JSON"))

def convert_path_to_wav(path, format, dest):
    import ffmpeg
    import pydub as pd
    pd.AudioSegment.ffmpeg = ffmpeg
    # pd.AudioSegment.converter = r"C:/Users/John/Downloads/ffmpeg-n5.1-latest-win64-gpl-5.1/ffmpeg-n5.1-latest-win64-gpl-5.1/bin/"
    # pd.AudioSegment.ffmpeg = r"C:/Users/John/Downloads/ffmpeg-n5.1-latest-win64-gpl-5.1/ffmpeg-n5.1-latest-win64-gpl-5.1/bin/"
    # pd.AudioSegment.ffprobe = r"C:/Users/John/Downloads/ffmpeg-n5.1-latest-win64-gpl-5.1/ffmpeg-n5.1-latest-win64-gpl-5.1/bin/"
    if type(path) == []: # Multiple Path merge
        pass
    else:
        files = os.listdir(path)
        
        i = len(os.listdir(os.path.join(PROJECT_PATH, dest)))
        for audio_file in files:
            try:
                print(os.path.join(path, audio_file))
                # pd.AudioSegment.from_file(
                #     file=os.path.join(path, audio_file), format=format).export(
                #         os.path.join(PROJECT_PATH, dest, str(i)+'.wav'), format='wav')
                    
                subprocess.call(['ffmpeg', '-i', str(os.path.join(path, audio_file)),str(os.path.join(PROJECT_PATH, dest, str(i)+'.wav'))])
                i+=1
            except Exception as err:
                print('Error:\n', err)
   
# print(convert_path_to_JSON(os.path.join('.','configurationState.JSON'), 'Spotify_Client_ID'))
# convert_path_to_wav(r"D:/OneDrive/GwenSmartHome/data/Users/TrainingSample/1/", 'w4a', 'data/SpeechModel/Training/1/')