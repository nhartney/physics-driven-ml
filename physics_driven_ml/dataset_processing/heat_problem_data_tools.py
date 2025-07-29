"""
The classes and methods needed for online training of the global heat problem example.
"""

import os
import numpy as np
import torch

from typing import NamedTuple, List, Optional
from firedrake import CheckpointFile, Function
from torch import Tensor
from firedrake.ml.pytorch import *
from torch.utils.data import Dataset, Subset

from collections import defaultdict


class BatchElement2(NamedTuple):
    """Batch element for PDE-based datasets as a tuple of PyTorch and Firedrake tensors."""
    u0: Tensor
    u_target: Tensor
    u0_fd: Function
    u_target_fd: Function


class BatchedElement2(NamedTuple):
    """Represent tensors for a list/batch of `BatchElement` that have been collated."""
    u0: Tensor  # shape = (batch_size, m)
    u_target: Tensor
    u0_fd: List[Function]
    u_target_fd: List[Function]
    batch_elements: Optional[List[BatchElement2]] = None


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


class PDEDataset2(Dataset):
    """
    Dataset reader for PDE-based datasets generated from the global heat example problem.
    """

    def __init__(self, dataset, data_dir):
        # Check dataset directory
        data = os.path.join(data_dir, dataset)
        if not os.path.exists(data):
            raise ValueError(f"Dataset directory {os.path.abspath(data)} does not exist")

        # Get mesh and batch elements (Firedrake functions)
        mesh, batch_elements = self.load_dataset(data)
        self.mesh = mesh
        self.batch_elements_fd = batch_elements
        self.fs = self.batch_elements_fd[0][0].function_space()

    def load_dataset(self, fname):
        data = []
        # Load data
        with CheckpointFile(fname, "r") as afile:
            n = int(np.array(afile.h5pyfile["n"]))
            # Load mesh
            mesh = afile.load_mesh("mesh")
            # Load data
            for i in range(n):
                initial_u = afile.load_function(mesh, "initial_u", idx=i)
                target_u = afile.load_function(mesh, "target_u", idx=i)
                data.append((initial_u, target_u))
        return mesh, data

    def __len__(self):
        return len(self.batch_elements_fd)

    def __getitem__(self, idx):
        u0_fd, u_target_fd = self.batch_elements_fd[idx]
        # Convert Firedrake functions to PyTorch tensors
        u0, u_target = [to_torch(e) for e in [u0_fd, u_target_fd]]
        return BatchElement2(u0=u0, u_target=u_target,
                             u0_fd=u0_fd,
                             u_target_fd=u_target_fd)

    def collate(self, batch_elements):
        # Workaround to enable custom data types (e.g. firedrake.Function) in PyTorch dataloaders
        # See: https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        batch_size = len(batch_elements)
        n = max(e.u0.size(-1) for e in batch_elements)
        m = max(e.u_target.size(-1) for e in batch_elements)

        u0 = torch.zeros(batch_size, n, dtype=batch_elements[0].u0.dtype)
        u_target = torch.zeros(batch_size, m, dtype=batch_elements[0].u_target.dtype)
        u0_fd = []
        u_target_fd = []
        for i, e in enumerate(batch_elements):
            u0[i, :] = e.u0
            u_target[i, :] = e.u_target
            u0_fd.append(e.u0_fd)
            u_target_fd.append(e.u_target_fd)

        return BatchedElement2(u0=0, u_target=u_target,
                              u0_fd=u0_fd, u_target_fd=u_target_fd,
                              batch_elements=batch_elements)


def list_duplicates(labels):
    # this method returns a list of tuples of (label, [indices]) where [indices] is a list of where
    # the label occurs
    tally = defaultdict(list)
    for i, item in enumerate(labels):
        tally[item].append(i)
    return (indices for indices in tally.items() if len(indices) > 1)


def sub_sample_point_data(point_train_dataloader):
    labels = []
    for step_num, point_sample in enumerate(point_train_dataloader):
        point_label = point_sample[:, -1].item()
        labels.append(point_label)
    # define a list of lists of indices where all the labels match
    index_list = []
    for lables, indices in sorted(list_duplicates(labels)):
        index_list.append(indices)

    # use index_list to sub-sample the point data by labels
    subsets = []
    for l in index_list:
        subset = Subset(point_train_dataloader.dataset, l)
        subsets.append(subset)
    # return a list of datasets, where all examples in each dataset belongs to one global sample
    return subsets
