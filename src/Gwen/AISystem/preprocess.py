import librosa
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

Data_Path = "C:/Users/John/OneDrive/GwenSmartHome/data/Models/SpeechModel/Training/"


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=Data_Path):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def get_mel_image_from_int32(audio_32, max_len=40, n_mfcc=28):
    audio = audio_32.astype(np.float32, order='C') / 32768.0
    # wave = audio[::3]
    mfcc = librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=n_mfcc)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# convert file to wav2mfcc
# Mel-frequency cepstral coefficients
def audioDataTomfcc(sound, max_len=64, n_mfcc=72):
    byte_data, sr, sw = sound.get_wav_data(), sound.sample_rate, sound.sample_width
    data_s16 = np.frombuffer(byte_data, dtype=np.int32, count=len(byte_data)//2, offset=0)
    float_data = data_s16 * 0.5**15 
    wave = float_data[::3]
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc)
    if (max_len > mfcc.shape[1]):
        pass
        # pad_width = max_len - mfcc.shape[1]
        # mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def wav2mfcc(file_path, max_len=64, n_mfcc=72):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = np.array(wave[::3])
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def npy2mfcc(file_path, max_len=64, n_mfcc=72):
    npz_file = np.load(file_path)
    wave = npz_file['audio_data']
    sr = npz_file['sample_rate']
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def save_data_to_array(path, max_len=64, n_mfcc=72):
    labels, _, _ = get_labels(os.path.join(path, "Audio"))

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [os.path.join(path, "Audio", label, wavfile) for wavfile in os.listdir(os.path.join(path, "Audio", label))]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
        np.save(os.path.join(label + '.npy', mfcc_vectors))


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(Data_Path)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



def prepare_dataset(path=Data_Path):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=sr)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=Data_Path):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]
# prepare_dataset("C:/Users/John/OneDrive/GwenSmartHome/data/Models/SpeechModel/Training/")