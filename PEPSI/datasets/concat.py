import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


from PEPSI.datasets.utils import *


class ConcatDataset(Dataset):
    def __init__(self,dataset_list, probs=None):
        self.datasets = dataset_list
        self.probs = probs if probs else [1/len(self.datasets)] * len(self.datasets)

    def __getitem__(self, i):
        chosen_dataset = np.random.choice(self.datasets, 1, p=self.probs)[0]
        i = i % len(chosen_dataset)
        return chosen_dataset[i]

    def __len__(self):
        return  max(len(d) for d in self.datasets)