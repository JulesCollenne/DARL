import torch
import torch.optim as optim

from darl import DARLModel
from data import get_loader
from losses import compute_reconstruction_loss, compute_disentanglement_loss
from metrics import plot_losses


def main():
    params = {
        "num_epochs": 20,
        "img_size": 224,
        "latent_dim": 100,
        "num_classes": 2,
        "log_interval": 50,
        "recon_loss_weight": 1.0,
        "disent_loss_weight": 0.1,
    }
    training = DARL_Train(params)
    training.train()


class DARL_Train:

    def __init__(self, args):
        self.num_epochs = args["num_epochs"]
        self.img_size = args["img_size"]
        self.latent_dim = args["latent_dim"]
        self.num_classes = args["num_classes"]
        self.log_interval = args["log_interval"]
        self.recon_loss_weight = args["recon_loss_weight"]
        self.disent_loss_weight = args["disent_loss_weight"]

        self.model = DARLModel(3, self.latent_dim, self.num_classes, self.img_size)
        self.train_loader, self.val_loader, self.test_loader = get_loader()

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        losses = []
        epoch_loss = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()

                reconstructed, disentangled = self.model(data)

                loss = self.compute_loss(reconstructed, data, disentangled)

                loss.backward()
                optimizer.step()

                if batch_idx % self.log_interval == 0:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.item()))

            epoch_loss /= len(self.train_loader)
            losses.append(epoch_loss)
            self.model.eval()
            with torch.no_grad():
                self.evaluate()

        torch.save(self.model.state_dict(), 'darl_model.pth')
        plot_losses(losses, "darl_losses.png")

        # In this example, we assume you have a DARLModel class that defines your DARL model and provides the forward method to compute the reconstructed and disentangled outputs. You can customize the loss functions (compute_reconstruction_loss and compute_disentanglement_loss) according to the specific disentanglement objectives and techniques you are using.
        #
        # The training loop iterates over the epochs and batches of the training dataset. For each batch, it performs the forward pass, computes the reconstruction loss and disentanglement loss, and combines them into a total loss using the specified weights. The backward pass is then performed to compute the gradients, and the optimizer is used to update the model parameters.
        #
        # Remember to adjust the hyperparameters, such as learning rate, number of epochs, and batch size, according to your specific requirements.

    def evaluate(self):
        """
        Evaluate the DARL model on the validation dataset.

        Args:
            model: The DARL model.
            val_loader: DataLoader for the validation dataset.

        Returns:
            float: Evaluation metric (e.g., accuracy, loss).
        """
        self.model.eval()  # Set the model to evaluation mode

        # Initialize variables for evaluation metric calculation
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch

                reconstructed, disentangled = self.model(inputs)

                loss = self.compute_loss(reconstructed, inputs, disentangled)

                batch_size = inputs.size(0)

                # Accumulate the total loss
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # Calculate the average evaluation metric
        average_loss = total_loss / total_samples
        return average_loss

    def compute_loss(self, reconstructed, inputs, disentangled):
        recon_loss = compute_reconstruction_loss(reconstructed, inputs)
        disent_loss = compute_disentanglement_loss(disentangled)
        return self.recon_loss_weight * recon_loss + self.disent_loss_weight * disent_loss


if __name__ == "__main__":
    main()
