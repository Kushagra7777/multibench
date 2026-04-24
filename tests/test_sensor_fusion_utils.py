import torch

from fusions.sensor_fusion import (
    roboticsConcat,
    sample_gaussian,
    product_of_experts,
    duplicate,
    depth_deconv,
)


def test_robotics_concat_default():
    x = [
        (torch.randn(2, 4), None),
        torch.randn(2, 4),
        torch.randn(2, 4),
        (torch.randn(2, 4), None),
        torch.randn(2, 4),
    ]
    out = roboticsConcat()(x)
    assert out.shape == (2, 20)


def test_robotics_concat_noconcat():
    x = [
        (torch.randn(2, 4), None),
        torch.randn(2, 4),
        torch.randn(2, 4),
        (torch.randn(2, 4), None),
        torch.randn(2, 4),
    ]
    out = roboticsConcat(name="noconcat")(x)
    assert isinstance(out, list)
    assert len(out) == 5


def test_robotics_concat_image():
    x = [
        (torch.randn(2, 4), None),
        (torch.randn(2, 4), None),
        torch.randn(2, 8),
    ]
    out = roboticsConcat(name="image")(x)
    assert out.shape == (2, 16)


def test_robotics_concat_simple():
    x = [torch.randn(2, 4), torch.randn(2, 6)]
    out = roboticsConcat(name="simple")(x)
    assert out.shape == (2, 10)


def test_sample_gaussian_shape():
    m = torch.zeros(4, 8)
    v = torch.ones(4, 8)
    z = sample_gaussian(m, v, torch.device('cpu'))
    assert z.shape == (4, 8)


def test_sample_gaussian_near_mean():
    torch.manual_seed(0)
    m = torch.full((1000, 1), 5.0)
    v = torch.full((1000, 1), 0.0001)
    z = sample_gaussian(m, v, torch.device('cpu'))
    assert abs(z.mean().item() - 5.0) < 0.1


def test_product_of_experts_shape():
    m = torch.randn(4, 8, 3)
    v = torch.ones(4, 8, 3)
    mu, var = product_of_experts(m, v)
    assert mu.shape == (4, 8)
    assert var.shape == (4, 8)


def test_product_of_experts_uniform_variance():
    m = torch.zeros(2, 4, 3)
    v = torch.ones(2, 4, 3)
    mu, var = product_of_experts(m, v)
    assert torch.allclose(mu, torch.zeros(2, 4))
    assert torch.allclose(var, torch.full((2, 4), 1.0 / 3))


def test_duplicate_shape():
    x = torch.tensor([[1.0, 2.0]])
    out = duplicate(x, 3)
    assert out.shape == (3, 2)


def test_duplicate_values():
    x = torch.tensor([[1.0, 2.0]])
    out = duplicate(x, 4)
    assert torch.allclose(out, x.expand(4, -1))


def test_depth_deconv_channels():
    layer = depth_deconv(16, 8)
    x = torch.randn(1, 16, 32, 32)
    out = layer(x)
    assert out.shape[1] == 8


def test_depth_deconv_upsamples():
    layer = depth_deconv(8, 4)
    x = torch.randn(1, 8, 16, 16)
    out = layer(x)
    assert out.shape[2] == 32
    assert out.shape[3] == 32
