import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image


class CatandDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        The is a dataset class, used to read image and label
        :param root_dir: the csv file path
        :param transform: if you want to use the data transform, transform is not None
        """
        self.annotations = pd.read_csv(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        return image, label
