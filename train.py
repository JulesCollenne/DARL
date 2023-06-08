from os.path import join

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn

from adversarial_model import AdversarialModel
from darl import DARLModel
from data import get_loader
from losses import compute_reconstruction_loss, compute_information_bottleneck_loss
from metrics import plot_losses


def main():
    params = {
        "num_epochs": 20,
        "img_size": 224,
        "batch_size": 4,
        "latent_dim": 200,
        "num_classes": 2,
        "log_interval": 50,
        "recon_loss_weight": 1.0,
        "disent_loss_weight": 0.1,
    }
    training = DarlTrain(params)
    training.train()


class DarlTrain:

    def __init__(self, args):
        self.num_epochs = args["num_epochs"]
        self.img_size = args["img_size"]
        self.latent_dim = args["latent_dim"]
        self.num_classes = args["num_classes"]
        self.log_interval = args["log_interval"]
        self.recon_loss_weight = args["recon_loss_weight"]
        self.disent_loss_weight = args["disent_loss_weight"]
        self.batch_size = args["batch_size"]

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available.")

        self.model = DARLModel(3, self.latent_dim, self.num_classes, self.img_size)
        self.model = self.model.to(self.device)
        self.adv_model = AdversarialModel(3, self.num_classes, self.img_size)
        self.adv_model = self.adv_model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader = get_loader(batch_size=self.batch_size)

    def train(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        adv_optimizer = optim.Adam(self.adv_model.parameters(), lr=0.001)
        adv_criterion = nn.CrossEntropyLoss()

        self.real_labels = torch.ones(self.batch_size).type(torch.LongTensor)
        self.fake_labels = torch.zeros(self.batch_size).type(torch.LongTensor)

        losses = []
        epoch_loss = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()
            self.adv_model.train()

            print("Training the discriminator...")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                z, reconstructed, logits = self.model(data)

                # ------------
                # Train the discriminator
                # ------------

                adv_optimizer.zero_grad()

                real_outputs = self.adv_model(data)
                real_loss = adv_criterion(real_outputs, self.real_labels.to(self.device))

                fake_outputs = self.adv_model(reconstructed)
                fake_loss = adv_criterion(fake_outputs, self.fake_labels.to(self.device))

                adv_loss = real_loss + fake_loss
                adv_loss.backward()
                adv_optimizer.step()

                if batch_idx % self.log_interval == 0:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tAdv Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), adv_loss.item()))
                    if adv_loss.item() < 1e-5:
                        break

            print("Training the generator...")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # ------------
                # Train the generator
                # ------------

                data = data.to(self.device)
                target = target.to(self.device)
                z, reconstructed, logits = self.model(data)

                optimizer.zero_grad()

                autoencoder_loss = self.compute_loss(z, reconstructed, logits, data, target)
                autoencoder_loss.backward()
                optimizer.step()

                # losses.append(total_loss.item())
                # epoch_loss += total_loss.item()

                if batch_idx % self.log_interval == 0:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tAE Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), autoencoder_loss.item()))
                    denormalized_recons = reconstructed.cpu().detach().numpy()[0] * \
                                          np.expand_dims(np.asarray(std), axis=(1, 2)) + \
                                          np.expand_dims(np.asarray(mean), axis=(1, 2))
                    plt.imshow(denormalized_recons.transpose(1, 2, 0))
                    plt.savefig(join("images", f"{batch_idx}.png"))

                if 100. * batch_idx / len(self.train_loader) > 10.:
                    break

            epoch_loss /= len(self.train_loader)
            losses.append(epoch_loss)
            # self.model.eval()
            # with torch.no_grad():
            #     self.evaluate()

        torch.save(self.model.state_dict(), 'darl_model.pth')
        plot_losses(losses, "darl_losses.png")

        # In this example, we assume you have a DARLModel class that defines your DARL model and provides the forward
        # method to compute the reconstructed and disentangled outputs. You can customize the loss functions (
        # compute_reconstruction_loss and compute_disentanglement_loss) according to the specific disentanglement
        # objectives and techniques you are using.
        #
        # The training loop iterates over the epochs and batches of the training dataset. For each batch, it performs
        # the forward pass, computes the reconstruction loss and disentanglement loss, and combines them into a total
        # loss using the specified weights. The backward pass is then performed to compute the gradients,
        # and the optimizer is used to update the model parameters.
        #
        # Remember to adjust the hyperparameters, such as learning rate, number of epochs, and batch size, according
        # to your specific requirements.

    def evaluate(self):
        """
        Evaluate the DARL model on the validation dataset.

        Returns:
            float: Evaluation metric (e.g., accuracy, loss).
        """
        self.model.eval()  # Set the model to evaluation mode

        # Initialize variables for evaluation metric calculation
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                z, reconstructed, logits = self.model(data)

                loss = self.compute_loss(z, reconstructed, logits, data, target).to(self.device)

                # Accumulate the total loss
                total_loss += loss.item() * self.batch_size
                total_samples += self.batch_size

        # Calculate the average evaluation metric
        average_loss = total_loss / total_samples
        return average_loss

    def compute_loss(self, z, reconstructed, logits, original_img, target,
                     recon_loss_weight=1.0, class_loss_weight=0.5, disentangle_loss_weight=0.2,
                     adv_loss_weight=0.5):
        # Compute the reconstruction loss
        recon_loss = compute_reconstruction_loss(reconstructed, original_img)

        # Compute the classification loss
        class_loss = nn.CrossEntropyLoss()(logits, target)

        # disentangle_loss = nn.MSELoss(z, labels)
        self.batch_size = z.size()[0]
        self.real_labels = torch.ones(self.batch_size).type(torch.LongTensor)
        disentangle_loss = compute_information_bottleneck_loss(z, self.batch_size)

        adv_loss = nn.CrossEntropyLoss()(self.adv_model(reconstructed), self.real_labels.to(self.device))

        # Compute the total loss as a combination of reconstruction loss and classification loss
        total_loss = recon_loss_weight * recon_loss + \
                     class_loss_weight * class_loss + \
                     disentangle_loss_weight * disentangle_loss + \
                     adv_loss * adv_loss_weight

        return total_loss

    # def compute_loss(self, reconstructed, inputs, disentangled):
    #     recon_loss = compute_reconstruction_loss(reconstructed, inputs)
    #     disent_loss = compute_disentanglement_loss(disentangled)
    #     class_loss = nn.CrossEntropyLoss()(classification, target_labels)
    #     return self.recon_loss_weight * recon_loss + self.disent_loss_weight * disent_loss


if __name__ == "__main__":
    main()
