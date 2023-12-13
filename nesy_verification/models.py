import torch
import torch.nn as nn


class SimpleEventCNN(nn.Module):
    def __init__(self, num_classes, log_softmax: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (3, 3))
        self.conv2 = nn.Conv2d(8, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.dense = nn.Linear(in_features=32, out_features=num_classes)
        self.log_softmax = log_softmax
        if log_softmax:
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, input_image):
        x = self.avg_pool(self.relu(self.conv1(input_image)))
        x = self.avg_pool(self.relu(self.conv2(x)))
        x = self.avg_pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.dense(x)

        magnitude_outputs = self.softmax(x[:, :3])
        parity_outputs = self.softmax(x[:, 3:])

        return torch.cat((magnitude_outputs, parity_outputs), dim=1)


class SimpleEventCNNnoSoftmax(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (3, 3))
        self.conv2 = nn.Conv2d(8, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.dense = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, input_image):
        x = self.avg_pool(self.relu(self.conv1(input_image)))
        x = self.avg_pool(self.relu(self.conv2(x)))
        x = self.avg_pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.dense(x)

        return x
