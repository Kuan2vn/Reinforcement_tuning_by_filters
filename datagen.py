# file: data_generator.py

import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset, DataLoader

# SPEC_PATH = 'archive_4'
IMG_SIZE = (128, 87)
label_encoder = {'Acoustic_guitar': 0, 'Applause': 1, 'Bark': 2, 'Bass_drum': 3, 'Burping_or_eructation': 4, 'Bus': 5, 'Cello': 6, 'Chime': 7, 'Clarinet': 8, 'Computer_keyboard': 9, 'Cough': 10, 'Cowbell': 11, 'Double_bass': 12, 'Drawer_open_or_close': 13, 'Electric_piano': 14, 'Fart': 15, 'Finger_snapping': 16, 'Fireworks': 17, 'Flute': 18, 'Glockenspiel': 19, 'Gong': 20, 'Gunshot_or_gunfire': 21, 'Harmonica': 22, 'Hi-hat': 23, 'Keys_jangling': 24, 'Knock': 25, 'Laughter': 26, 'Meow': 27, 'Microwave_oven': 28, 'Oboe': 29, 'Saxophone': 30, 'Scissors': 31, 'Shatter': 32, 'Snare_drum': 33, 'Squeak': 34, 'Tambourine': 35, 'Tearing': 36, 'Telephone': 37, 'Trumpet': 38, 'Violin_or_fiddle': 39, 'Writing': 40}

# Define constants
IMG_SIZE = (128, 87)
SPEC_PATH = 'spectrograms'

class AudioDataset(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train

    def __getitem__(self, index):
        FILE = self.dataframe.fname.values[index]
        LABEL = self.dataframe.label.values[index]

        # Try loading from the specified set (train or test)
        SET = 'train_spec' if self.is_train else 'test_spec'
        path = f'{SPEC_PATH}/{SET}/{FILE[:-4]}.npy'
        
        try:
            data_array = np.load(path)
        except FileNotFoundError:
            # If not found, switch to the other set
            SET = 'test_spec' if self.is_train else 'train_spec'
            path = f'{SPEC_PATH}/{SET}/{FILE[:-4]}.npy'
            try:
                data_array = np.load(path)
            except FileNotFoundError:
                print(f"Error: File not found in either train_spec or test_spec: {path}")
                return None, None

        resized = cv2.resize(data_array, (IMG_SIZE[1], IMG_SIZE[0]))
        X = np.zeros(shape=(3, IMG_SIZE[0], IMG_SIZE[1]))
        for j in range(3):
            X[j,:,:] = resized

        if self.is_train:
            y = label_encoder[LABEL]  # Assuming you have a label encoder defined
            return torch.tensor(X, dtype=torch.float), y
        else:
            return torch.tensor(X, dtype=torch.float)

    def __len__(self):
        return self.dataframe.shape[0]

