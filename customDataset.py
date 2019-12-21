import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class DeepFakeSmallDataset(Dataset):
    """Real Fake images small sample dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with binary labels (real: 1, fake: 0).
            root_dir (string): Directory with all the real and fake images after applying FFT (DFT) to faces in images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.root_dir, self.labels.iloc[idx, 1]))
        label = self.labels.iloc[idx, 2]
        
        return self.transform(image), label