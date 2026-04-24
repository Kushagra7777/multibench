import torch
import torch.nn as nn

from unimodals.robotics.layers import (
    crop_like,
    deconv,
    predict_flow,
    conv2d,
    View,
    Flatten,
    CausalConv1D,
    ResidualBlock,
)
from unimodals.robotics.models_utils import init_weights, rescaleImage, filter_depth


def test_crop_like_same_size():
    inp = torch.randn(1, 3, 4, 4)
    target = torch.randn(1, 3, 4, 4)
    out = crop_like(inp, target)
    assert out.shape == inp.shape


def test_crop_like_different_size():
    inp = torch.randn(1, 3, 8, 8)
    target = torch.randn(1, 3, 4, 4)
    out = crop_like(inp, target)
    assert out.shape == (1, 3, 4, 4)


def test_deconv_output_channels():
    layer = deconv(16, 8)
    out = layer(torch.randn(1, 16, 4, 4))
    assert out.shape[1] == 8


def test_deconv_upsamples():
    layer = deconv(8, 4)
    out = layer(torch.randn(1, 8, 4, 4))
    assert out.shape[2] == 8 and out.shape[3] == 8


def test_predict_flow_output():
    layer = predict_flow(16)
    out = layer(torch.randn(1, 16, 4, 4))
    assert out.shape == (1, 2, 4, 4)


def test_conv2d_same_spatial():
    layer = conv2d(16, 8)
    out = layer(torch.randn(1, 16, 4, 4))
    assert out.shape == (1, 8, 4, 4)


def test_conv2d_with_dilation():
    layer = conv2d(8, 4, kernel_size=3, dilation=2)
    out = layer(torch.randn(1, 8, 8, 8))
    assert out.shape == (1, 4, 8, 8)


def test_view_module():
    view = View((-1, 8))
    out = view(torch.randn(2, 4, 2))
    assert out.shape == (2, 8)


def test_flatten_module():
    flat = Flatten()
    out = flat(torch.randn(2, 3, 4, 4))
    assert out.shape == (2, 48)


def test_causal_conv1d_output_length():
    conv = CausalConv1D(4, 8, kernel_size=3)
    out = conv(torch.randn(1, 4, 10))
    assert out.shape == (1, 8, 10)


def test_causal_conv1d_kernel_size_1():
    conv = CausalConv1D(4, 8, kernel_size=1)
    out = conv(torch.randn(1, 4, 10))
    assert out.shape == (1, 8, 10)


def test_residual_block_shape():
    block = ResidualBlock(16)
    out = block(torch.randn(1, 16, 8, 8))
    assert out.shape == (1, 16, 8, 8)


def test_rescale_image_shape():
    image = torch.randint(0, 255, (2, 4, 6, 3)).float()
    out = rescaleImage(image)
    assert out.shape == (2, 3, 4, 6)


def test_rescale_image_values():
    image = torch.full((1, 4, 4, 3), 255.0)
    out = rescaleImage(image)
    assert abs(out.max().item() - 1.0) < 1e-5


def test_filter_depth_zeros_small():
    depth = torch.tensor([[0.0, 1e-8, 0.5, 1.5]])
    out = filter_depth(depth)
    assert out[0, 0].item() == 0.0
    assert out[0, 1].item() == 0.0
    assert abs(out[0, 2].item() - 0.5) < 1e-6


def test_filter_depth_zeros_large():
    depth = torch.tensor([[2.5, 1.0]])
    out = filter_depth(depth)
    assert out[0, 0].item() == 0.0
    assert abs(out[0, 1].item() - 1.0) < 1e-6


def test_init_weights_runs():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 3, 3),
    )
    init_weights(model.children())
