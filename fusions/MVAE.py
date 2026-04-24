"""Implements MVAE."""
import torch
from torch import nn


class ProductOfExperts(nn.Module):
    """
    Return parameters for product of independent experts.

    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """

    def __init__(self, size):
        """Initialize Product of Experts Object.

        Args:
            size (tuple): Size of Product of Experts Layer

        """
        super(ProductOfExperts, self).__init__()
        self.size = size

    def forward(self, mus, logvars, eps=1e-8):
        """Apply Product of Experts Layer.

        Args:
            mus (torch.Tensor): torch.Tensor of Mus
            logvars (torch.Tensor): torch.Tensor of Logvars
            eps (float, optional): Epsilon for log-exponent trick. Defaults to 1e-8.

        Returns:
            torch.Tensor, torch.Tensor: Output of PoE layer.
        """
        device = mus[0].device
        mu, logvar = _prior_expert(self.size, len(mus[0]), device)
        mu = torch.cat([mu] + [m.unsqueeze(0) for m in mus], dim=0)
        logvar = torch.cat([logvar] + [lv.unsqueeze(0) for lv in logvars], dim=0)

        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class ProductOfExperts_Zipped(nn.Module):
    """
    Return parameters for product of independent experts.
    
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """

    def __init__(self, size):
        """Initialize Product of Experts Object.

        Args:
            size (tuple): Size of Product of Experts Layer
        
        """
        super(ProductOfExperts_Zipped, self).__init__()
        self.size = size

    def forward(self, zipped, eps=1e-8):
        """Apply Product of Experts Layer.

        Args:
            mus (torch.Tensor): torch.Tensor of Mus
            logvars (torch.Tensor): torch.Tensor of Logvars
            eps (float, optional): Epsilon for log-exponent trick. Defaults to 1e-8.

        Returns:
            torch.Tensor, torch.Tensor: Output of PoE layer.
        """
        mus = [i[0] for i in zipped]
        logvars = [i[1] for i in zipped]
        device = mus[0].device
        mu, logvar = _prior_expert(self.size, len(mus[0]), device)
        mu = torch.cat([mu] + [m.unsqueeze(0) for m in mus], dim=0)
        logvar = torch.cat([logvar] + [lv.unsqueeze(0) for lv in logvars], dim=0)

        var = torch.exp(logvar) + eps
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


def _prior_expert(size, batch_size, device):
    """Universal prior expert — spherical Gaussian N(0, 1)."""
    size = (size[0], batch_size, size[2])
    mu = torch.zeros(size, device=device)
    logvar = torch.zeros(size, device=device)  # log(1) == 0
    return mu, logvar
