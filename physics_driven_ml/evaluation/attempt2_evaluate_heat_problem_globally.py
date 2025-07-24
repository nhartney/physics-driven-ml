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
from physics_driven_ml.dataset_processing import PointDataset, BatchedElement

from train_heat_problem_globally import sub_sample_point_data
from attempt2_train_heat_problem_globally import calculate_global_loss


def evaluate_globally_by_point(model, point_dl, global_dl):
    """
    Evaluate the model on a given dataset.
    Compute the L2 error of the NN for every sample in the evaluation set, and then
    add these errors up to give an overall error for that model. (The 'sample' in the
    evaluation set is a global example, made up of all points in the domain.)
    """

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
            loss = calculate_global_loss(subset, model)
            total_error += loss

        if step_num == eval_steps - 1:
            break

    # L2 error is the square root of the total error
    L2_error = total_error**0.5
    L2_error /= eval_steps
    return L2_error