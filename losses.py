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


def compute_information_bottleneck_loss(z, batch_size):
    """
    Computes the Information Bottleneck (IB) loss for promoting disentangled features in the penultimate layer.

    Args:
        z (torch.Tensor): The penultimate layer output tensor of shape (batch_size, num_features).
        batch_size (int): The size of the current batch.

    Returns:
        torch.Tensor: The IB loss tensor.

    """

    # Compute the covariance matrix
    covariance = torch.matmul(z.t(), z) / batch_size

    # Compute the inverse of the covariance matrix
    inverse_covariance = torch.inverse(covariance)

    # Compute the log determinant of the covariance matrix
    log_det_covariance = torch.logdet(covariance)

    # Compute the IB loss using the trace of the product between the inverse covariance and the covariance matrix
    ib_loss = torch.trace(torch.matmul(inverse_covariance, covariance)) - log_det_covariance

    return ib_loss
