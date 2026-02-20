# NOTE dataloader needs to be implemented
#   Please use special/kinetics_gradient_blend.py for now
import sys
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torchvision

sys.path.append(os.getcwd())

from unimodals.common_models import ResNetLSTMEnc, MLP
from datasets.kinetics.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test



filename = 'best3.pt'
traindata, validdata, testdata = get_dataloader(sys.argv[1])
r50 = torchvision.models.resnet50(pretrained=True)
r50.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
audio_encoder = torch.nn.Sequential(r50, MLP(1000, 200, 64)).to(device)
encoders = [ResNetLSTMEnc(64).to(device), audio_encoder.to(device)]
mult_head = MLP(64+64, 200, 5).to(device)
uni_head = [MLP(64, 200, 5).to(device), MLP(64, 200, 5).to(device)]

fusion = Concat().to(device)

train(encoders, mult_head, uni_head, fusion, traindata, validdata, 300,
      gb_epoch=10, optimtype=torch.optim.SGD, lr=0.01, savedir=filename)

print("Testing:")
model = torch.load(filename, weights_only=False).to(device)
test(model, testdata)
