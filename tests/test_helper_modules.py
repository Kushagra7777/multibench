import torch
import torch.nn as nn

from utils.helper_modules import Sequential2


def test_sequential2_shape():
    seq = Sequential2(nn.Linear(10, 5), nn.ReLU())
    out = seq(torch.randn(4, 10))
    assert out.shape == (4, 5)


def test_sequential2_identity():
    seq = Sequential2(nn.Identity(), nn.Identity())
    x = torch.randn(3, 7)
    assert torch.allclose(seq(x), x)


def test_sequential2_nonnegative_after_relu():
    seq = Sequential2(nn.Linear(4, 4), nn.ReLU())
    out = seq(torch.randn(8, 4))
    assert (out >= 0).all()
