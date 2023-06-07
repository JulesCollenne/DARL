import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialModel(nn.Module):
    def __init__(self, input_dim, latent_dim, img_size):
        super(AdversarialModel, self).__init__()
        num_downsamples = 3
        output_size = img_size // (2 ** num_downsamples)

        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(128 * output_size * output_size, latent_dim)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        x = self.relu3(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = torch.sigmoid(x)
        return x
