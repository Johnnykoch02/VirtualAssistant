#Imports
import os
import subprocess as sp
import json
import sys
import numpy as np
from datetime import timedelta
import librosa

### convert_path_to_JSON(path, ['password', 'username', 'api_key'])
PLAYREAD_SPEED = 48000

def get_json_variables(path, var_name):
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        json_file.close()
    if type(var_name) is list:
        return_data = {}
        for key in var_name:
            return_data[key] = data[key]
        return return_data
    else:            
        return {var_name: data[var_name]}
# path= os.path.abspath()
# print(os.path.join(path,"configurationState.JSON"))

def get_mel_image_from_int32(audio_32, max_len=40, n_mfcc=28, sound_rate=PLAYREAD_SPEED):
    audio = audio_32.astype(np.float32, order='C') / 32768.0
    # wave = audio[::3]
    mfcc = librosa.feature.mfcc(y=audio, sr=sound_rate, n_mfcc=n_mfcc)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc
def normalize_mfcc(mfcc, min_val=-1, max_val=1):
    mfcc_min, mfcc_max = mfcc.min(), mfcc.max()
    return ((mfcc - mfcc_min) * (max_val - min_val) / (mfcc_max - mfcc_min) + min_val).astype(np.float32)
        
def get_mel_image_from_float_normalized(audio_float, max_len=40, n_mfcc=28, sound_rate=PLAYREAD_SPEED):
    audio = audio_float.astype(np.float32, order='C') * 1.0
    mfcc = librosa.feature.mfcc(y=audio, sr=sound_rate, n_mfcc=n_mfcc)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return np.array(mfcc)

def convert_path_to_wav(path, dest):
    if type(path) == []: # Multiple Path merge
        pass
    else:
        files = os.listdir(path)
        
        i = len(os.listdir(os.path.join(os.getcwd(), dest)))
        for audio_file in files:
            try:
                print(os.path.join(path, audio_file))
                sp.call(['ffmpeg', '-i', str(os.path.join(path, audio_file)),str(os.path.join(os.getcwd(), dest, str(i)+'.wav'))])
                i+=1
            except Exception as err:
                print('Error:\n', err)
                
def GetAudioData(filename, secondSampleSize):
    command = [ 'ffmpeg',
            '-i', filename,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(PLAYREAD_SPEED), 
            '-ac', '2', # stereo (set to '1' for mono)
            '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    lengthCalculation = (44100 * secondSampleSize) * 4
    return pipe.stdout.read(int(lengthCalculation))        #The raw audio

def GetAudioDataSegment(filename, currentTime, sampleSize):

    command = [ 'ffmpeg',
            '-i', filename,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(PLAYREAD_SPEED), # ouput will have 44100 Hz
            '-ac', '2', # stereo (set to '1' for mono)
            '-ss', str(timedelta(seconds=currentTime)),
            '-to', str(timedelta(seconds=currentTime+sampleSize)),
            '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

    lengthCalculation = ((44100 * sampleSize) * 4)

    return pipe.stdout.read(int(lengthCalculation)) # The raw audio

def get_length(filename):
    result = sp.run(['ffprobe', "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
        stdout=sp.PIPE,
        stderr=sp.STDOUT)
    return float(result.stdout)

def SpliceAudioData(file_name, sampleSize):
    allSegments = []
    print(file_name)
    master_time = int(get_length(file_name) / sampleSize)
    print(master_time)
    for i in range(0, master_time):
        raw_audio = GetAudioDataSegment(file_name, i * sampleSize, sampleSize)
        allSegments.append(np.frombuffer(raw_audio, dtype="int32"))

    return allSegments

def GetAudioSplice(audioData, dataSamplesPerSplice, index):
    return audioData[index * dataSamplesPerSplice : (index + 1) * dataSamplesPerSplice]          
    
   
# print(convert_path_to_JSON(os.path.join('.','configurationState.JSON'), 'Spotify_Client_ID'))
# convert_path_to_wav(r"D:/OneDrive/GwenSmartHome/data/Users/TrainingSample/1/", 'w4a', 'data/SpeechModel/Training/1/')