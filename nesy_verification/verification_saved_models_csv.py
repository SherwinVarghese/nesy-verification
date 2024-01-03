"""Provide verification bounds for the saved models"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_utils import MNISTSimpleEvents
from models import SimpleEventCNN, SimpleEventCNNnoSoftmax
from torch.utils.data import DataLoader, random_split

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

NUM_MAGNITUDE_CLASSES = 3
NUM_PARITY_CLASSES = 2
PRINT = True
NUM_SAMPLES = 10

SAVED_MODELS_PATH = "nesy-verification/nesy_verification/saved_models/icl"
VERIFICATION_RESULTS_PATH = "nesy-verification/nesy_verification/results/verification"

def bound_softmax(h_L, h_U):
    """Given lower and upper input bounds into a softmax, calculate their concrete output bounds."""

    shift = h_U.max(dim=1, keepdim=True).values
    exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
    lower = exp_L / (torch.sum(exp_U, dim=1, keepdim=True) - exp_U + exp_L + 1e-7)
    upper = exp_U / (torch.sum(exp_L, dim=1, keepdim=True) - exp_L + exp_U + 1e-7)
    return lower, upper


def calculate_bounds(model: torch.nn.Module, dataloader):
    """Calculate bounds for the provided model.

    Note that there is a magnitude classification task (num < 3, 3 < num < 6,
    num > 6) and a parity classification task, i.e. (even(num), odd(num))

    Args:
        is_magnitude_classification
    """

    epsilons = [0.5, 0.1, 0.01]
    soft_max = torch.nn.Softmax(dim=1)
    for eps in epsilons:
        num_samples_verified = 0
        num_samples_correctly_classified = 0
        num_symbols_safe = 0


        for dl_idx, (image_ids, test_inputs, test_classes) in enumerate(dataloader):
            if torch.cuda.is_available():
                test_inputs = test_inputs.cuda()
                test_classes = test_classes.cuda()
                model = model.cuda()

            # wrap model with auto_LiRPA
            lirpa_model = BoundedModule(
                model,
                torch.empty_like(test_inputs),
                device=test_inputs.device,
                verbose=True,
            )
            # print("Running on", test_inputs.device)

            # compute bounds for the final output

            ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
            ptb_test_inputs = BoundedTensor(test_inputs, ptb)

            pred = lirpa_model(ptb_test_inputs)
            pred_classes = (pred > 0.0).float()  # Predicted labels

            for method in [
                "IBP",
                "IBP+backward",
                "CROWN-Optimized"
            ]:
                print(f"Using the bounding method: {method}")
                
                veri_results = list()

                lb, ub = lirpa_model.compute_bounds(x=(ptb_test_inputs,), method=method)

                # First handle magnitude verification
                for i in range(len(test_classes)):
                    num_samples_verified += 1

                    if PRINT:
                        print(
                            f"Image id: {image_ids[i]} top-1 prediction is: {pred_classes[i]}, the ground-truth is: {test_classes[i]}"
                        )

                    if torch.allclose(
                        pred_classes[i], test_classes[i]
                    ):  # if the tensors are approximately similar
                        num_samples_correctly_classified += 1

                        # Verification bounds for each image is stored in ver_res
                        ver_res = []
                        ver_res.append(image_ids[i].item()) # Adding image Id to ver_res
                        
                        lb_softmax, ub_softmax = bound_softmax(lb, ub)

                        for truth_idx in range(len(test_classes[i])):
                            if (lb[i][truth_idx] > 0).item():
                                num_symbols_safe += 1
                            
                            # Adding the lower and upper bounds to ver_res
                            ver_res.append(lb_softmax[i][truth_idx].item()) 
                            ver_res.append(ub_softmax[i][truth_idx].item()) 
            
                        # Appending results
                        veri_results.append(ver_res)
                        

                print(f"For the method: {method}")
                print(f"Num samples verified: {num_samples_verified}")
                print(f"Num samples correctly classified: {num_samples_correctly_classified}")
                print(f"Num symbols safe: {num_symbols_safe}")
                print(f"Total symbols: {num_samples_verified * 5}")
                print()
                print(f"Saving verification results for {eps} as CSV")
                df = pd.DataFrame(veri_results, columns = [
                                    "Image ID",
                                    "lower_bound_n<3",
                                    "upper_bound_n<3",
                                    "lower_bound_3<=n<=6",
                                    "upper_bound_3<=n<=6",
                                    "lower_bound_>6",
                                    "upper_bound_>6",
                                    "lower_bound_%2==0",
                                    "upper_bound_%2==0",
                                    "lower_bound_%2!=0",
                                    "upper_bound_%2!=0",
                                ])
                df.to_csv(os.path.join(VERIFICATION_RESULTS_PATH, "verification_cnn_"+method+"_"+str(eps)+".csv"), index=False)
                
            # if dl_idx == NUM_SAMPLES:
            #     break    


if __name__ == "__main__":
    saved_models_path = os.path.join(
        # Path(__file__).parent.resolve(), "/saved_models/icl"
        SAVED_MODELS_PATH
    )

    # Load the Log Softmax model
    cnn_with_softmax = SimpleEventCNN(num_classes=5, log_softmax=False)
    cnn_with_softmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_with_log_softmax.pt"))
    )

    # Load the Softmax model
    cnn_with_logsoftmax = SimpleEventCNN(num_classes=5, log_softmax=True)
    cnn_with_logsoftmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_with_softmax.pt"))
    )

    # Load the Non Softmax model
    cnn_no_softmax = SimpleEventCNNnoSoftmax(num_classes=5)
    cnn_no_softmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_with_no_softmax.pt"))
    )

    # Getting test data
    dataset = MNISTSimpleEvents()
    train_indices = torch.load(os.path.join(saved_models_path, "train_indices.pt"))
    test_indices = torch.load(os.path.join(saved_models_path, "test_indices.pt"))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    N = len(dataset)

    test_dl = DataLoader(test_dataset, batch_size=N)

    cnn_with_softmax.eval()

    print("Verifying CNN Softmax")
    # Epoch 50/50 	---	 loss (train): 0.0001	 loss (test): 0.1261	 f1_magnitude (test): 0.9858	 f1_parity (test): 0.9900
    calculate_bounds(cnn_no_softmax, test_dl)
    print("--------------------------------------------------------")
