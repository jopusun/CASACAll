import torch
import os
from torch.utils.data import Dataset
import numpy as np
class BasecallDataset(Dataset):
    def __init__(self, signal_dir, refrence_dir):
        file_list = os.listdir(signal_dir)
        self.file_count = len(file_list)
        self.file_list = file_list
        self.signal_path = signal_dir
        self.label_path = refrence_dir
        self.signals = list()
        self.labels = list()
        self.idx = 0
    def __len__(self):
        return self.file_count
    def __getitem__(self, index):
        signal = np.load()