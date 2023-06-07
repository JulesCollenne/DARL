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


def compute_information_bottleneck_loss(z, batch_size, eps=1e-6):
    """
    Computes the Information Bottleneck (IB) loss for promoting disentangled features in the penultimate layer.

    Args:
        eps:  a small epsilon (eps) is added when computing the inverse and the log determinant to prevent division by
        zero and to handle numerical stability.
        z (torch.Tensor): The penultimate layer output tensor of shape (batch_size, num_features).
        batch_size (int): The size of the current batch.

    Returns:
        torch.Tensor: The IB loss tensor.

    """

    # Compute the covariance matrix
    covariance = torch.matmul(z.t(), z) / batch_size

    # Compute the singular value decomposition (SVD) of the covariance matrix
    u, s, v = torch.svd(covariance)

    # Compute the pseudo-inverse of the singular values
    pseudo_inverse_s = torch.diag(1.0 / torch.sqrt(s + eps))

    # Compute the inverse covariance matrix using SVD
    inverse_covariance = torch.matmul(torch.matmul(v, pseudo_inverse_s), u.t())

    # Compute the log determinant of the covariance matrix using SVD
    log_det_covariance = torch.sum(torch.log(s + eps))

    # Compute the IB loss using the trace of the product between the inverse covariance and the covariance matrix
    ib_loss = torch.trace(torch.matmul(inverse_covariance, covariance)) - log_det_covariance

    return ib_loss


__all__ = ['kl', 'reconstruction', 'discriminator_logistic_simple_gp',
           'discriminator_gradient_penalty', 'generator_logistic_non_saturating']


def kl(mu, log_var):
    return -0.5 * torch.mean(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), 1))


def reconstruction(recon_x, x, lod=None):
    return torch.mean((recon_x - x) ** 2)


def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=10.0):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def discriminator_gradient_penalty(d_result_real, reals, r1_gamma=10.0):
    real_loss = d_result_real.sum()
    real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
    loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()
