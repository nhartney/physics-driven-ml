import os
import numpy as np
import torch

from typing import List
from firedrake import CheckpointFile
from firedrake.ml.pytorch import *
from torch.utils.data import Dataset

from physics_driven_ml.dataset_processing import BatchElement2, BatchedElement2


class PDEDataset2(Dataset):
    """
    Dataset reader for PDE-based datasets generated from the global heat example problem.
    """

    def __init__(self, dataset, dataset_split, data_dir):
        print("we're in the init of PDEDataset2")
        # Check dataset directory
        dataset_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")

        # Get mesh and batch elements (Firedrake functions)
        name_file = dataset_split + "_global_data.h5"
        mesh, batch_elements = self.load_dataset(os.path.join(dataset_dir, name_file))
        self.mesh = mesh
        self.batch_elements_fd = batch_elements


    def load_dataset(self, fname):
        print("we're in the load_dataset of PDEDataset2")
        data = []
        # Load data
        with CheckpointFile(fname, "r") as afile:
            n = int(np.array(afile.h5pyfile["n"]))
            # Load mesh
            mesh = afile.load_mesh("mesh")
            # Load data
            for i in range(n):
                target_f = afile.load_function(mesh, "target_f", idx=i)
                data.append((target_f))
        return mesh, data


    def __len__(self):
        print("we're in the len of PDEDataset2")
        return len(self.batch_elements_fd)


    def __getitem__(self, idx):
        print("we're in the get_item of PDEDataset2")
        target_fd = self.batch_elements_fd[idx]
        # Convert Firedrake functions to PyTorch tensors
        target = [to_torch(e) for e in [target_fd]]
        return BatchElement2(target=target, target_fd=target_fd)


    def collate(self, batch_elements):
        print("using custom collate function")
        # Workaround to enable custom data types (e.g. firedrake.Function) in PyTorch dataloaders
        # See: https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        batch_size = len(batch_elements)
        n = max(e.u_obs.size(-1) for e in batch_elements)
        m = max(e.target.size(-1) for e in batch_elements)

        target = torch.zeros(batch_size, m, dtype=batch_elements[0].target.dtype)
        target_fd = []
        for i, e in enumerate(batch_elements):
            target[i, :] = e.target
            target_fd.append(e.target_fd)

        return BatchedElement2(target=target,
                              target_fd=target_fd,
                              batch_elements=batch_elements)

