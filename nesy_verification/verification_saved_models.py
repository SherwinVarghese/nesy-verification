"""Provide verification bounds for the saved models"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch.utils.data import DataLoader, random_split

from data_utils import MNISTSimpleEvents
from models import SimpleEventCNN, SimpleEventCNNnoSoftmax


NUM_MAGNITUDE_CLASSES = 3
NUM_PARITY_CLASSES = 2
PRINT = True
NUM_SAMPLES = 20

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

    epsilons = [0.01]
    for eps in epsilons:

        num_magnitude_samples_verified = 0
        num_magnitude_samples_correctly_classified = 0
        num_magnitude_samples_safe = 0
        num_parity_samples_correctly_classified = 0

        for dl_idx, (test_inputs, test_labels) in enumerate(dataloader):

            magnitude_test_labels = torch.argmax(test_labels[:, :3], dim=1)
            parity_test_labels = torch.argmax(test_labels[:, 3:], dim=1)


            if torch.cuda.is_available():
                test_inputs = test_inputs.cuda()
                magnitude_test_labels = magnitude_test_labels.cuda()
                parity_test_labels = parity_test_labels.cuda()
                model = model.cuda()

            # wrap model with auto_LiRPA
            lirpa_model = BoundedModule(model, torch.empty_like(test_inputs), device=test_inputs.device, verbose=True)
            # print("Running on", test_inputs.device)

            # compute bounds for the final output

            ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
            ptb_test_inputs = BoundedTensor(test_inputs, ptb)

            pred = lirpa_model(ptb_test_inputs)
            magnitude_pred_labels = torch.argmax(pred[:, :3], dim=1)
            parity_pred_labels = torch.argmax(pred[:, 3:], dim=1)

            for method in [
                "IBP",
            ]:
                # print(f"Using the bounding method: {method}")

                lb, ub = lirpa_model.compute_bounds(x=(ptb_test_inputs,), method=method.split()[0])

                # First handle magnitude verification
                for i in range(len(magnitude_test_labels)):
                    num_magnitude_samples_verified += 1

                    # print(
                    #     f"Image {i} top-1 prediction is: {magnitude_pred_labels[i]}, the ground-truth is: {magnitude_test_labels[i]}"
                    # )

                    if magnitude_pred_labels[i] == magnitude_test_labels[i]:
                        num_magnitude_samples_correctly_classified += 1

                        lb_magnitude, ub_magnitude = bound_softmax(lb[:, :3], ub[:, :3])  # TODO EdS: Why is this not the same as having a softmax output

                        truth_idx = int(magnitude_test_labels[i])

                        if (lb_magnitude[:, :3][i][truth_idx] > torch.cat((ub_magnitude[:, :3][i][:truth_idx], ub_magnitude[:, :3][i][truth_idx + 1:]))).all().item():
                            safe = True
                            num_magnitude_samples_safe += 1
                        else:
                            safe = False

                        if PRINT:
                            for j in range(NUM_MAGNITUDE_CLASSES):
                                indicator = "(ground-truth)" if j == magnitude_test_labels[i] else ""
                                pred_indicator = "(prediction)" if j == magnitude_pred_labels[i] else ""
                                safe_indicator = "(safe)" if j == magnitude_test_labels[i] and safe else ""
                                print(
                                    "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind} {pred} {safe}".format(
                                        j=j, l=lb_magnitude[i][j].item(), u=ub_magnitude[i][j].item(), ind=indicator, pred=pred_indicator, safe=safe_indicator
                                    )
                                )

                            # if (lb_magnitude[:, :3][0][truth_idx] > torch.cat((ub_magnitude[:, :3][0][:truth_idx], ub_magnitude[:, :3][0][truth_idx + 1:]))).all().item():
                            #     num_magnitude_samples_safe += 1
                            x = 1

                print()

            # if dl_idx == NUM_SAMPLES:
            #     break
            # Then handle parity verification
            # for i in range(len(magnitude_test_labels)):
            #     print(
            #         f"Image {i} top-1 prediction is: {magnitude_pred_labels[i]}, the ground-truth is: {magnitude_test_labels[i]}"
            #     )
            #
            #     if magnitude_pred_labels[i] == magnitude_test_labels[i]:
            #         num_magnitude_samples_correctly_classified += 1
            #
            #         lb_magnitude, ub_magnitude = bound_softmax(lb[:, :3], ub[:,
            #                                                               :3])  # TODO EdS: Why is this not the same as having a softmax output
            #
            #         for j in range(NUM_MAGNITUDE_CLASSES):
            #             indicator = "(ground-truth)" if j == magnitude_test_labels[i] else ""
            #             print(
            #                 "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
            #                     j=j, l=lb_magnitude[i][j].item(), u=ub_magnitude[i][j].item(), ind=indicator
            #                 )
            #             )
            #
            #             # TODO EdS: Now determine if they're safe
            #             num_magnitude_samples_safe

                # TODO EdS: Now do parity lb_parity, ub_parity = bound_softmax(lb[:, 3:], ub[:, 3:])

        print(f"----\nSUMMARY\n----")
        print(f"For the method: {method}")
        print(f"Num magnitude samples verified: {num_magnitude_samples_verified}")
        print(f"Num magnitude samples correctly classified: {num_magnitude_samples_correctly_classified}")
        print(f"Num magnitude samples safe: {num_magnitude_samples_safe}")
        print()


if __name__ == "__main__":
    saved_models_path = os.path.join(
        # Path(__file__).parent.resolve(), "/saved_models/icl"
        "/Users/edward/github/nesy-verification/nesy_verification/saved_models/icl"
    )

    # Load the Softmax model
    cnn_with_softmax = SimpleEventCNN(num_classes=5, log_softmax=False)
    cnn_with_softmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_with_log_softmax.pt"))
    )

    # Load the Log Softmax model
    cnn_with_logsoftmax = SimpleEventCNN(num_classes=5, log_softmax=True)
    cnn_with_logsoftmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_with_softmax.pt"))
    )

    # Load the Log Softmax model
    cnn_no_softmax = SimpleEventCNNnoSoftmax(num_classes=5)
    cnn_no_softmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_with_no_softmax.pt"))
    )

    # Getting test data
    dataset = MNISTSimpleEvents()
    train_indices = torch.load(os.path.join(saved_models_path, 'train_indices.pt'))
    test_indices = torch.load(os.path.join(saved_models_path, 'test_indices.pt'))
    # dummy_indices = torch.load(os.path.join(os.getcwd(), 'saved_models/dummy_indices.pt'))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    # dummy_dataset = torch.utils.data.Subset(dataset, dummy_indices)
    N = len(dataset)

    test_dl = DataLoader(test_dataset, batch_size=32)

    cnn_with_softmax.eval()

    # print("Verifying CNN LogSoftmax for Magnitude Classification ")
    # Epoch 50/50 	---	 loss (train): 0.0000	 loss (test): 0.1678	 f1_magnitude (test): 0.9827	 f1_parity (test): 0.9899
    # calculate_bounds(cnn_with_logsoftmax, is_magnitude_classification=True)
    # print("--------------------------------------------------------")

    # print("Verifying CNN Softmax for Magnitude Classification ")
    # Epoch 50/50 	---	 loss (train): -1.9809	 loss (test): -1.9593	 f1_magnitude (test): 0.9776	 f1_parity (test): 0.9825
    # calculate_bounds(cnn_with_softmax, is_magnitude_classification=True)
    # print("--------------------------------------------------------")

    print("Verifying CNN Softmax")
    # Epoch 50/50 	---	 loss (train): 0.0001	 loss (test): 0.1261	 f1_magnitude (test): 0.9858	 f1_parity (test): 0.9900
    calculate_bounds(cnn_no_softmax, test_dl)
    print("--------------------------------------------------------")
