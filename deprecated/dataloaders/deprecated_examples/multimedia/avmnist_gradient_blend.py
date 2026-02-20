from unimodals.common_models import LeNet, MLP, Constant
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test
import sys
import os
sys.path.append(os.getcwd())

filename = 'best3.pt'
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
mult_head = MLP(channels*40, 100, 10).to(device)
uni_head = [MLP(channels*8, 100, 10).to(device), MLP(channels*32, 100, 10).to(device)]

fusion = Concat().to(device)

train(encoders, mult_head, uni_head, fusion, traindata, validdata, 300,
      gb_epoch=10, optimtype=torch.optim.SGD, lr=0.01, savedir=filename)

print("Testing:")
model = torch.load(filename, weights_only=False).to(device)
test(model, testdata)
