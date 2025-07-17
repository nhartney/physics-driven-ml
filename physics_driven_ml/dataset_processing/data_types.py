from typing import NamedTuple, List, Optional
from firedrake import Function
from torch import Tensor


class BatchElement(NamedTuple):
    """Batch element for PDE-based datasets as a tuple of PyTorch and Firedrake tensors."""
    u_obs: Tensor  # shape = (n,)
    target: Tensor  # shape = (m,)
    u_obs_fd: Function
    target_fd: Function


class BatchedElement(NamedTuple):
    """Represent tensors for a list/batch of `BatchElement` that have been collated."""
    u_obs: Tensor  # shape = (batch_size, n)
    target: Tensor  # shape = (batch_size, m)
    u_obs_fd: List[Function]
    target_fd: List[Function]
    batch_elements: Optional[List[BatchElement]] = None


class PointBatchElement(NamedTuple):
    """Batch element for point data-based datasets as a tuple of Pytorch tensors."""
    u: Tensor
    f: Tensor
    x: Tensor
    y: Tensor
    t: Tensor


class PointBatchedElement(NamedTuple):
    """Represent tensors for a list of `PointBatchElement' that have been collated."""
    u: Tensor
    f: Tensor
    x: Tensor
    y: Tensor
    t: Tensor
    batch_elements: Optional[List[BatchElement]] = None


class BatchElement2(NamedTuple):
    """Batch element for PDE-based datasets as a tuple of PyTorch and Firedrake tensors."""
    target: Tensor  # shape = (m,)
    target_fd: Function


class BatchedElement2(NamedTuple):
    """Represent tensors for a list/batch of `BatchElement` that have been collated."""
    target: Tensor  # shape = (batch_size, m)
    target_fd: List[Function]
    batch_elements: Optional[List[BatchElement]] = None