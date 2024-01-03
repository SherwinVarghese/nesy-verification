"""Provide verification bounds for the saved models"""
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST

from models import SimpleEventCNN

BATCH_SIZE = 32
NUM_MAGNITUDE_CLASSES = 3
NUM_PARITY_CLASSES = 2
PRINT = False
NUM_SAMPLES = 20
MODEL_PATH = Path(__file__).parent.resolve() / "saved_models/icl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pd.options.display.float_format = "{:.10f}".format

RESULTS_COLUMNS = [
    "mnist_id",
    "magnitude_0_lb",
    "magnitude_0_ub",
    "magnitude_1_lb",
    "magnitude_1_ub",
    "magnitude_2_lb",
    "magnitude_2_ub",
    "parity_0_lb",
    "parity_0_ub",
    "parity_1_lb",
    "parity_1_ub",
    "magnitude_prediction_idx",
    "magnitude_label_idx",
    "parity_prediction_idx",
    "parity_label_idx",
    "magnitude_correct",
    "parity_correct",
    "magnitude_safe",
    "parity_safe",
    "safe",
]


def bound_softmax(h_L, h_U):
    """Given lower and upper input bounds into a softmax, calculate their concrete
    output bounds."""

    shift = h_U.max(dim=1, keepdim=True).values
    exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
    lower = exp_L / (torch.sum(exp_U, dim=1, keepdim=True) - exp_U + exp_L + 1e-7)
    upper = exp_U / (torch.sum(exp_L, dim=1, keepdim=True) - exp_L + exp_U + 1e-7)
    return lower, upper


def calculate_bounds(model: torch.nn.Module, dataloader, epsilon: float, method: str = "IBP"):
    """Calculate bounds for the provided model and dataset.

    Note that there is a magnitude classification task (num < 3, 3 < num < 6,
    num > 6) and a parity classification task, i.e. (even(num), odd(num))
    """

    print(f"Performing verification with an epsilon of {epsilon}")
    print(f"Using the bounding method: {method}") if PRINT else None

    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_magnitude_samples_verified = 0
    num_magnitude_samples_correctly_classified = 0
    num_magnitude_samples_safe = 0
    num_parity_samples_verified = 0
    num_parity_samples_correctly_classified = 0
    num_parity_samples_safe = 0
    num_samples_safe = 0

    for dl_idx, (mnist_idx, inputs, labels) in enumerate(dataloader):
        print(f"Starting new batch") if PRINT else None

        magnitude_labels = torch.argmax(labels[:, :3], dim=1)
        parity_labels = torch.argmax(labels[:, 3:], dim=1)

        if dl_idx == 0:
            if DEVICE == "cuda":
                inputs = inputs.cuda()
                magnitude_labels = magnitude_labels.cuda()
                parity_labels = parity_labels.cuda()
                model = model.cuda()
            print("Running on", DEVICE)

            lirpa_model = BoundedModule(
                model,
                torch.empty_like(inputs),
                device=inputs.device,
                verbose=True,
            )

        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        ptb_inputs = BoundedTensor(inputs, ptb)

        pred = lirpa_model(ptb_inputs)
        magnitude_preds = torch.argmax(pred[:, :3], dim=1)
        parity_preds = torch.argmax(pred[:, 3:], dim=1)

        lb, ub = lirpa_model.compute_bounds(
            x=(ptb_inputs,), method=method.split()[0]
        )

        # First handle magnitude verification
        for i in range(len(magnitude_labels)):
            new_row = {"mnist_id": mnist_idx[i].item()}

            # First check magnitude
            num_magnitude_samples_verified += 1

            print(
                f"Image {i} top-1 prediction is: {magnitude_preds[i]}, the "
                f"ground-truth is: {magnitude_labels[i]}"
            ) if PRINT else None

            if magnitude_preds[i] == magnitude_labels[i]:
                num_magnitude_samples_correctly_classified += 1

                # Pass the bounds through a softmax bounding layer
                # TODO EdS: A sanity check is required to check why I was getting
                #  different results when using a softmax layer in auto_LiRPA
                lb_magnitude, ub_magnitude = bound_softmax(lb[:, :3], ub[:, :3])

                truth_idx = int(magnitude_labels[i])

                # Check that the lower bound of the truth class is greater than
                # the upper bound of all other classes
                if (
                    (
                        lb_magnitude[:, :3][i][truth_idx]
                        > torch.cat(
                            (
                                ub_magnitude[:, :3][i][:truth_idx],
                                ub_magnitude[:, :3][i][truth_idx + 1 :],
                            )
                        )
                    )
                    .all()
                    .item()
                ):
                    magnitude_safe = True
                    num_magnitude_samples_safe += 1
                else:
                    magnitude_safe = False

                for j in range(NUM_MAGNITUDE_CLASSES):
                    indicator = "(ground-truth)" if j == magnitude_labels[i] else ""
                    pred_indicator = (
                        "(prediction)" if j == magnitude_preds[i] else ""
                    )
                    safe_indicator = (
                        "(safe)" if j == magnitude_labels[i] and magnitude_safe else ""
                    )
                    print(
                        "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind} {pred} {safe}".format(
                            j=j,
                            l=lb_magnitude[i][j].item(),
                            u=ub_magnitude[i][j].item(),
                            ind=indicator,
                            pred=pred_indicator,
                            safe=safe_indicator,
                        )
                    ) if PRINT else None
                    new_row[f"magnitude_{j}_lb"] = lb_magnitude[i][j].item()
                    new_row[f"magnitude_{j}_ub"] = ub_magnitude[i][j].item()
                    new_row["magnitude_correct"] = True
                    new_row["magnitude_prediction_idx"] = magnitude_preds[i].item()
                    new_row["magnitude_label_idx"] = magnitude_labels[i].item()
                    new_row["magnitude_safe"] = magnitude_safe  # TODI EdS: DRY

            else:
                # The magnitude has not been correctly predicted:
                new_row[f"magnitude_0_lb"] = None
                new_row[f"magnitude_0_ub"] = None
                new_row[f"magnitude_1_lb"] = None
                new_row[f"magnitude_1_ub"] = None
                new_row[f"magnitude_2_lb"] = None
                new_row[f"magnitude_2_ub"] = None
                new_row["magnitude_prediction_idx"] = magnitude_preds[i].item()
                new_row["magnitude_label_idx"] = magnitude_labels[i].item()
                new_row["magnitude_correct"] = False
                magnitude_safe = False
                new_row["magnitude_safe"] = magnitude_safe

            # Second check parity
            num_parity_samples_verified += 1

            print(
                f"Image {i} top-1 parity prediction is: {parity_preds[i]}, "
                f"the ground-truth is: {parity_labels[i]}"
            ) if PRINT else None

            if parity_preds[i] == parity_labels[i]:
                num_parity_samples_correctly_classified += 1

                # Pass the bounds through a softmax bounding layer
                # TODO EdS: A sanity check is required to check why I was getting
                #  different results when using a softmax layer in auto_LiRPA
                lb_parity, ub_parity = bound_softmax(lb[:, 3:], ub[:, 3:])

                truth_idx = int(parity_labels[i])

                # Check that the lower bound of the truth class is greater than
                # the upper bound of all other classes
                if (
                        (
                                lb_parity[:, :3][i][truth_idx]
                                > torch.cat(
                            (
                                    ub_parity[:, :3][i][:truth_idx],
                                    ub_parity[:, :3][i][truth_idx + 1:],
                            )
                        )
                        )
                                .all()
                                .item()
                ):
                    parity_safe = True
                    num_parity_samples_safe += 1
                else:
                    parity_safe = False

                for j in range(NUM_PARITY_CLASSES):
                    indicator = "(ground-truth)" if j == parity_labels[i] else ""
                    pred_indicator = (
                        "(prediction)" if j == parity_preds[i] else ""
                    )
                    safe_indicator = (
                        "(safe)" if j == parity_labels[i] and parity_safe else ""
                    )
                    print(
                        "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind} {pred} {safe}".format(
                            j=j,
                            l=lb_parity[i][j].item(),
                            u=ub_parity[i][j].item(),
                            ind=indicator,
                            pred=pred_indicator,
                            safe=safe_indicator,
                        )
                    ) if PRINT else None
                    new_row[f"parity_{j}_lb"] = lb_parity[i][j].item()
                    new_row[f"parity_{j}_ub"] = ub_parity[i][j].item()
                    new_row["parity_correct"] = True
                    new_row["parity_prediction_idx"] = parity_preds[i].item()
                    new_row["parity_label_idx"] = parity_labels[i].item()
                    new_row["parity_safe"] = parity_safe

            else:
                # The parity has not been correctly predicted:
                new_row[f"parity_0_lb"] = None
                new_row[f"parity_0_ub"] = None
                new_row[f"parity_1_lb"] = None
                new_row[f"parity_1_ub"] = None
                new_row["parity_prediction_idx"] = parity_preds[i].item()
                new_row["parity_label_idx"] = parity_labels[i].item()
                new_row["parity_correct"] = False
                parity_safe = False
                new_row["parity_safe"] = parity_safe

            safe = True if magnitude_safe and parity_safe else False
            new_row["safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    # Quick sanity check
    assert num_parity_samples_verified == num_magnitude_samples_verified

    print(f"----\nSUMMARY\n----")
    print(f"For the method: {method}")
    print(f"Num magnitude samples verified: {num_magnitude_samples_verified}")
    print(f"Num magnitude samples correctly classified: {num_magnitude_samples_correctly_classified}")
    print(f"Num magnitude samples safe: {num_magnitude_samples_safe}")
    print(f"Num parity samples verified: {num_parity_samples_verified}")
    print(f"Num parity samples correctly classified: {num_parity_samples_correctly_classified}")
    print(f"Num parity samples safe: {num_parity_samples_safe}")
    print()

    results_summary = {
        "method": f"{method}",
        "num_magnitude_samples_verified": f"{num_magnitude_samples_verified}",
        "num_magnitude_samples_correctly_classified": f"{num_magnitude_samples_correctly_classified}",
        "num_magnitude_samples_safe": f"{num_magnitude_samples_safe}",
        "num_parity_samples_verified": f"{num_parity_samples_verified}",
        "num_parity_samples_correctly_classified": f"{num_parity_samples_correctly_classified}",
        "num_parity_samples_safe": f"{num_parity_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    save_results_to_csv(df_results, results_summary, epsilon)


def save_results_to_csv(results: pd.DataFrame, summary: dict, epsilon=None):
    results.to_csv(
        MODEL_PATH / f"results_{epsilon}.csv",
        index=False,
    )

    with open(
        MODEL_PATH / f"summary_{epsilon}.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=4)


def load_model(model_filename: str, num_classes: int):
    model = SimpleEventCNN(num_classes=num_classes, log_softmax=True)
    model.load_state_dict(torch.load(MODEL_PATH / model_filename))
    return model


def load_datasets() -> Tuple:
    dataset = MNIST(os.path.expanduser("~/.cache/mnist"), train=True, download=True)

    train_indices = torch.load(MODEL_PATH / "train_indices.pt")
    test_indices = torch.load(MODEL_PATH / "test_indices.pt")
    train_subset = CustomSubset(dataset, train_indices)
    test_subset = CustomSubset(dataset, test_indices)

    train_dl = DataLoader([d for d in train_subset], batch_size=BATCH_SIZE)
    test_dl = DataLoader([d for d in test_subset], batch_size=BATCH_SIZE)
    return train_dl, test_dl


class CustomSubset(Subset):
    """Created a custom subset to have access to MNIST indices when creating the CSV."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.transform = transforms.PILToTensor()
        self.normalise = transforms.Compose(
            [
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def classify_label(self, x):
        """Given the MNIST digit, return the class label (for magnitude classification
        then parity classification)."""
        return [
            x < 3,
            3 <= x <= 6,
            x > 6,
            x % 2 == 0,
            x % 2 != 0,
        ]

    def __getitem__(self, idx):
        # one_hot_tensor = torch.nn.functional.one_hot(torch.tensor(self.dataset[self.indices[idx]][1]), NUM_MAGNITUDE_CLASSES)
        return (
            self.indices[idx],
            self.normalise(self.transform(self.dataset[self.indices[idx]][0]).float()),
            torch.tensor(
                self.classify_label(self.dataset[self.indices[idx]][1])
            ).float(),
        )


class CustomDataset(Dataset):
    """Created a custom subset to have access to MNIST indices when creating the CSV."""

    def __init__(self, data, indices):
        super().__init__(data, indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.dataset[self.indices[idx]]


if __name__ == "__main__":
    cnn_no_softmax = load_model("cnn_no_softmax.pt", 5)

    train_dl, test_dl = load_datasets()

    cnn_no_softmax.eval()

    print("Verifying CNN Softmax")
    calculate_bounds(cnn_no_softmax, test_dl, epsilon=0.1)

    # TODO: Robustify the model to get smaller bounds
