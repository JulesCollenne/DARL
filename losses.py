import torch
import torch.nn.functional as F


def compute_reconstruction_loss(reconstructed, data):
    """
    Compute the reconstruction loss between reconstructed images and original input data.

    Args:
        reconstructed (torch.Tensor): Reconstructed images.
        data (torch.Tensor): Original input data.

    Returns:
        torch.Tensor: Reconstruction loss.
    """
    reconstruction_loss = F.mse_loss(reconstructed, data)
    return reconstruction_loss


def compute_disentanglement_loss(disentangled):
    """
    Compute the disentanglement loss based on the disentangled representations.

    Args:
        disentangled (torch.Tensor): Disentangled representations.

    Returns:
        torch.Tensor: Disentanglement loss.
    """
    # Compute the disentanglement loss based on the desired objectives
    # and specific constraints of your disentanglement objectives.

    # Example: Compute the L1 norm of the disentangled representations
    disentanglement_loss = torch.mean(torch.abs(disentangled))

    return disentanglement_loss
