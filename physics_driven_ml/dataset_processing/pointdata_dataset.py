import os
import numpy as np
import torch

from typing import List
from firedrake.ml.pytorch import *
from torch.utils.data import Dataset


class PointDataset(Dataset):
    """
    Dataset reader for data generated point-wise from a PDE solution. The pointwise data should be
    saved as numpy arrays.
    """

    def __init__(self, numpy_data, data_dir):
       # Check dataset directory
        dataset_dir = os.path.join(data_dir, "datasets", numpy_data)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")
        self.numpy_list = np.load(numpy_data)


    def __len__(self):
        return len(self.numpy_list)


    def __getitem__(self, idx):
        # Make the sample a numpy array first
        numpy_sample = np.array(self.numpy_list[idx])
        # Convert the numpy array to a PyTorch tensor
        tensor_sample = torch.from_numpy(numpy_sample)
        return tensor_sample


