import os

import torch
import firedrake as fd
import firedrake.ml.pytorch as fd_ml

from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from physics_driven_ml.models import PointNN
from physics_driven_ml.utils import get_logger
from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2, BatchedElement2

from physics_driven_ml.training.train_heat_problem_globally import sub_sample_point_data

from firedrake.output import VTKFile

# from firedrake import exp
import numpy as np


def evaluate_globally(model, device, point_dl, global_dl, disable_tqdm=False, write_out_results=True):
    """
    Evaluate the model on a given dataset.
    Compute the L2 error of the NN for every sample in the evaluation set, and then
    add these errors up to give an overall error for that model. (The 'sample' in the
    evaluation set is a global example, made up of all points in the domain.)
    """

    batch_size = 1
    device = device

    model.eval()

    eval_steps = len(global_dl)
    total_error = 0.0

    point_data_subsets = sub_sample_point_data(point_dl)

    if len(point_data_subsets) != eval_steps:
        print("The number of data subsets does not match the number of global samples")
 
    
    for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_data_subsets, global_dl))),
                                                    total=eval_steps, disable=disable_tqdm):

        # Move batch to device
        batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])

        # Exact solution as a function
        target_fd, = batch.target_fd

        # Do a forward pass on all points in the data subset and accumulate loss
        with torch.no_grad():

            # Do a forward pass on all points in the data subset
            global_network_pred = forward_pass(subset, model, batch_size)

            # convert this to a Firedrake function
            network_pred_fd = fd_ml.from_torch(global_network_pred, target_fd.function_space())

            total_error += eval_error(network_pred_fd, target_fd)

            if write_out_results:
                # output both the prediction and the target to a vtu file
                # access the label on the data for saving outputting
                label = subset[-1][-1].item()

                outfile = VTKFile(f'evaluation/evaluation_plots/test_plot_{label}.pvd')
                outfile.write(network_pred_fd, target_fd)

        if step_num == eval_steps - 1:
            break
    
    total_error /= eval_steps
    return total_error


def forward_pass(point_train_data_subset, model, batch_size):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. It
    builds a list of network predictions at each point (tensors), which are then concatenated into one
    tensor to give a global network estimate of f, which is what is returned by the method.
    """

    tensor_list = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)

    for step_num, batch in enumerate(point_train_dl):

        # extract inputs and target from the tensor
        inputs = batch[:, 1:5]

        # forward pass
        network_point_f = model(inputs)[:,0]

        # add to list of tensor solutions
        tensor_list.append(network_point_f)

    # concatenate the tensor list together
    f_tensor = torch.cat(tuple(tensor_list))

    return f_tensor

def eval_error(x, x_exact):
    """Compute the L2 error between x and x_exact"""
    return fd.norm(x - x_exact)


if __name__ == "__main__":

    logger = get_logger("Evaluation")

    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/")
    dataset = "heat_problem_global_validation_data"
    batch_size = 1
    device = "cpu"
    evaluation_metric = "L2"
    model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models/heat_problem_globally/"
    model_version = "heat_problem_globally_epoch-19-error_0.11140"

    # Load dataset
    dataset_dir = os.path.join(data_dir, "datasets", dataset)
    logger.info(f"Loading dataset from {dataset_dir}\n")

    point_dataset = PointDataset(numpy_data=os.path.join(dataset_dir, "numpy_point_validate_data.npy"),
                           data_dir=data_dir)
    point_dataloader = DataLoader(point_dataset, batch_size=batch_size, shuffle=False)

    global_dataset = PDEDataset2(dataset=os.path.join(dataset_dir, "global_validate_data.h5"),
                                       data_dir=dataset_dir)
    global_dataloader = DataLoader(global_dataset, batch_size=batch_size,
                                 collate_fn=global_dataset.collate, shuffle=False)

    # Load model
    model_dir = os.path.join(model_dir, model_version)

    logger.info(f"Loading model checkpoint from {model_dir}\n")
    model = PointNN()
    # Load pretrained model state dict
    pretrained = torch.load(os.path.join(model_dir, "model.pt"))
    model.load_state_dict(pretrained)

     # Set double precision (default Firedrake type)
    model.double()
    # Move model to device
    model.to(device)

    # Evaluate model
    error = evaluate_globally(model, device, point_dataloader, global_dataloader)
    logger.info(f"\n\t Error (metric: {evaluation_metric}): {error:.4e}")