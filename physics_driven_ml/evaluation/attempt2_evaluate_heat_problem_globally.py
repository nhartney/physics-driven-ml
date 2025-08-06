import os
import argparse

import torch
import firedrake as fd
import firedrake.ml.pytorch as fd_ml

from torch.utils.data import DataLoader

from functools import partial
from tqdm.auto import tqdm

from physics_driven_ml.models import PointNN
from physics_driven_ml.utils import ModelConfig, get_logger
from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2, BatchedElement2

from physics_driven_ml.training.train_heat_problem_globally import sub_sample_point_data, calculate_global_loss

from firedrake.output import VTKFile

# from firedrake import exp
import numpy as np


def evaluate_globally_by_point(model, device, point_dl, global_dl, disable_tqdm=False, write_out_results=True):
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
        
        subset_dl = DataLoader(subset, batch_size=batch_size, shuffle=False)

        # Do a forward pass on all points in the data subset and accumulate loss
        with torch.no_grad():
            if write_out_results:
                # get network prediction for f as a Firedrake function
                loss, f_func = calculate_global_loss(subset, global_dl, model, output_loss_fn=True)
                # get target as a function from the global dataloader
                batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
                target_f = batch.target_fd[0]

                # output both the prediction and the target to a vtu file
                # access the label on the data for saving outputting
                label = subset[-1][-1].item()

                # next undo the transform on the target and the network's output
                # recovered_f = exp((-10/f) + 1e-10) - 1e-10
                # f_func.dat.data[:] = np.exp((-10/f_func.dat.data[:]) + 1e-10) - 1e-10
                # target_f.dat.data[:] = np.exp((-10/target_f.dat.data[:]) + 1e-10) - 1e-10

                outfile = VTKFile(f'evaluation/evaluation_plots/test_plot_{label}.pvd')
                outfile.write(target_f, f_func)
            else:
                loss = calculate_global_loss(subset, global_dl, model, output_loss_fn=False)
    
            total_error += loss

        if step_num == eval_steps - 1:
            break

    # L2 error is the square root of the total error
    L2_error = total_error**0.5
    L2_error /= eval_steps
    return L2_error


# def write_out_results(subset_dl, model):
#     # Output Firedrake function from the network's prediction for f and the target f function for
#     # visual comparison
#     for step_num, batch in enumerate(subset_dl):
#         # extract inputs and target from the tensor
#         inputs = batch[:, 1:5]
#         target_f = batch[:,0]
#         # forward pass
#         network_point_f = model(inputs)[:,0]
#         # compute L2 loss at the point
#         loss = l2_loss(network_point_f, target_f)
#         # add point loss to total loss list
#         loss_list.append(loss)


if __name__ == "__main__":

    logger = get_logger("Evaluation")

    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/")
    dataset = "heat_problem_global_validation_data"
    batch_size = 1
    device = "cpu"
    evaluation_metric = "L2"
    model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models/heat_problem_globally/"
    model_version = "heat_problem_globally_epoch-19-error_0.12404"

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
    error = evaluate_globally_by_point(model, device, point_dataloader, global_dataloader)
    logger.info(f"\n\t Error (metric: {evaluation_metric}): {error:.4e}")