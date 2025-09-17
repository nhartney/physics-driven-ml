"""
The classes and methods needed for offline and online training of the global heat problem
example.
"""

import os
import numpy as np

from numpy import random

import torch

from typing import NamedTuple, List, Optional
from firedrake import *
from firedrake.__future__ import interpolate
from torch import Tensor
from firedrake.ml.pytorch import *
from torch.utils.data import Dataset, Subset, DataLoader

from collections import defaultdict


class BatchElementOffline(NamedTuple):
    """Batch element for PDE-based datasets as a tuple of PyTorch and Firedrake tensors."""
    f_target: Tensor
    f_target_fd: Function


class BatchedElementOffline(NamedTuple):
    """Represent tensors for a list/batch of `BatchElement` that have been collated."""
    f_target: Tensor
    f_target_fd: List[Function]
    batch_elements: Optional[List[BatchElementOffline]] = None


class BatchElementOnline(NamedTuple):
    """Batch element for PDE-based datasets as a tuple of PyTorch and Firedrake tensors."""
    u0: Tensor
    u_target: Tensor
    u0_fd: Function
    u_target_fd: Function


class BatchedElementOnline(NamedTuple):
    """Represent tensors for a list/batch of `BatchElement` that have been collated."""
    u0: Tensor  # shape = (batch_size, m)
    u_target: Tensor
    u0_fd: List[Function]
    u_target_fd: List[Function]
    batch_elements: Optional[List[BatchElementOnline]] = None


class PointDataset(Dataset):
    """
    Dataset reader for data generated point-wise from a PDE solution. The pointwise data should be
    saved as numpy arrays.
    """

    def __init__(self, numpy_data, data_dir=None):
        # Check dataset directory
        if data_dir is not None:
            dataset_dir = os.path.join(data_dir, "datasets", numpy_data)
            if not os.path.exists(dataset_dir):
                raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")
            self.numpy_list = np.load(numpy_data)
        else:
            self.numpy_list = numpy_data

    def __len__(self):
        return len(self.numpy_list)

    def __getitem__(self, idx):
        # Make the sample a numpy array first
        numpy_sample = np.array(self.numpy_list[idx])
        # Convert the numpy array to a PyTorch tensor
        tensor_sample = torch.from_numpy(numpy_sample)
        return tensor_sample


class PDEDatasetOffline(Dataset):
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
        self.fs = self.batch_elements_fd[0].function_space()

    def load_dataset(self, fname):
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
        return len(self.batch_elements_fd)

    def __getitem__(self, idx):
        f_target_fd = self.batch_elements_fd[idx]
        # Convert Firedrake functions to PyTorch tensors
        f_target = to_torch(f_target_fd)
        return BatchElementOffline(f_target=f_target,
                                   f_target_fd=f_target_fd)

    def collate(self, batch_elements):
        # Workaround to enable custom data types (e.g. firedrake.Function) in PyTorch dataloaders
        # See: https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        batch_size = len(batch_elements)
        m = max(e.f_target.size(-1) for e in batch_elements)

        f_target = torch.zeros(batch_size, m, dtype=batch_elements[0].f_target.dtype)
        f_target_fd = []
        for i, e in enumerate(batch_elements):
            f_target[i, :] = e.f_target
            f_target_fd.append(e.f_target_fd)

        return BatchedElementOffline(f_target=f_target,
                                     f_target_fd=f_target_fd,
                                     batch_elements=batch_elements)


class PDEDatasetOnline(Dataset):
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
        return BatchElementOnline(u0=u0, u_target=u_target,
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

        return BatchedElementOnline(u0=0, u_target=u_target,
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


def forward_pass_by_point(point_train_data_subset, model, batch_size):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. It
    returns a list of network predictions at each point, which can then be interpolated to a Firedrake
    function for a global network estimate of f.
    """
    network_out = []
    ordered_coords = []
    point_train_dl = DataLoader(point_train_data_subset,
                                batch_size=batch_size, shuffle=False)
    for step_num, batch in enumerate(point_train_dl):
        # model.zero_grad()
        # extract inputs from the tensor
        inputs = batch[:, 1:5]
        # forward pass
        network_point_f = model(inputs)[:, 0]
        # extract value from the output tensor
        network_point_f_value = network_point_f.item()

        # this is just for debugging
        # print("this is the point value predicted by the network:", network_point_f_value)
        # target_point_f = batch[:, 0]
        # print("this is the target point f value:", target_point_f.item())

        # add value to list
        network_out.append(network_point_f_value)
        # add coordinate to coordinate list (same ordering as data list)
        x = batch[:, 3].item()
        y = batch[:, 4].item()
        ordered_coords.append((x, y))
    return np.asarray(network_out), ordered_coords


def interpolate_to_firedrake_function(global_dl_mesh, global_dl_fs, network_f, coordinates_list):

    mesh = global_dl_mesh
    # create vertex-only mesh with the same input ordering as the coordinates
    vom = VertexOnlyMesh(mesh, coordinates_list, redundant=False)

    # start with a function space that is structured like the data
    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    field_vomio = Function(P0DG_io)
    field_vomio.dat.data_wo[:] = network_f

    # Next interpolate onto the function space that does not have
    # the input ordering
    P0DG = FunctionSpace(vom, "DG", 0)
    field_vom = assemble(interpolate(field_vomio, P0DG))

    # interpolate from this VOM to the parent mesh (global data mesh)
    # find the function space that the global target function is on
    Vsrc = global_dl_fs

    I = Interpolator(TestFunction(Vsrc), P0DG)
    f_star = field_vom.riesz_representation(riesz_map="l2")
    f_data_star = Cofunction(Vsrc.dual())
    I.interpolate(f_star, adjoint=True, output=f_data_star)

    # Do this with the new interpolate behaviour
    # f_data_star = assemble(interpolate(f_star, P0DG, adjoint=True))

    field = f_data_star.riesz_representation(riesz_map="l2")
    return field


def generate_initial_conditions(mesh, n):
    ICs_list = []
    x, y = SpatialCoordinate(mesh)
    # Produce n random samples for initial conditions
    for r in range(n):
        x_pos = random.rand()
        y_pos = random.rand()
        a = 1 + random.rand()
        IC = a*exp(-((x-x_pos)**2)/0.01-((y-y_pos)**2)/0.01)
        ICs_list.append(IC)
    return ICs_list


def train_test_split(point_data_list, global_data_list, train_proportion):
    total_global_samples = len(global_data_list)
    total_point_samples = len(point_data_list)
    pp_sample = int(total_point_samples/total_global_samples)

    n_global_train = int(train_proportion*total_global_samples)
    n_global_test = int(total_global_samples - n_global_train)

    n_point_train = int(n_global_train*pp_sample)
    n_point_test = int(total_point_samples - n_point_train)

    point_train, point_test = point_data_list[:n_point_train], point_data_list[:n_point_test]

    global_train, global_test = global_data_list[:n_global_train], global_data_list[:n_global_test]

    # check that the labels for point and global data in each set match
    point_train_labels = []
    for p in point_train:
        label_p = p[-1]
        point_train_labels.append(label_p)
    global_train_labels = []
    for g in global_train:
        label_g = g[-1]
        global_train_labels.append(label_g)
    for l in point_train_labels:
        if l not in global_train_labels:
            raise Exception("The point data label has no corresponding global data label")

    return point_train, point_test, global_train, global_test