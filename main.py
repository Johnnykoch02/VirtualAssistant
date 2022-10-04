import os
PROJECT_PATH = os.getcwd()
import sys



def main():
    from src.utils import convert_path_to_wav
    convert_path_to_wav("C:\\Users\\John\\Documents\\Sound recordings", 'w4a', 'data/Models/SpeechModel/Training/0/')


if __name__ == '__main__':
    main()
    
    