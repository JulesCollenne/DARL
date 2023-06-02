import torch
import torch.nn as nn
import torch.nn.functional as F


class DARLModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, img_size):
        super(DARLModel, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, img_size)
        self.decoder = Decoder(latent_dim, input_dim, img_size)
        self.classifier = Classifier(latent_dim, num_classes)

    def forward(self, x):
        # Encode input
        z = self.encoder(x)

        # Reconstruct input
        reconstructed = self.decoder(z)

        # Classify latent representation
        logits = self.classifier(z)

        return reconstructed, logits


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, img_size):
        super(Encoder, self).__init__()
        num_downsamples = 2
        output_size = img_size // (2 ** num_downsamples)
        # self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.fc = nn.Linear(64 * output_size * output_size, latent_dim)

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
        # print(x.size())
        # x = F.relu(self.conv1(x))
        # print(x.size())
        # x = F.relu(self.conv2(x))
        # print(x.size())
        # x = x.view(x.size(0), -1)
        # print(x.size())
        # x = self.fc(x)
        # print(x.size())
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        x = self.relu3(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, img_size):
        super(Decoder, self).__init__()
        num_downsamples = 2
        output_size = img_size // (2 ** num_downsamples)
        self.fc = nn.Linear(latent_dim, 64 * output_size * output_size)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 8, 8)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
