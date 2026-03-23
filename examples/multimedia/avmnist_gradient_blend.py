import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test
from torch.utils.data import DataLoader, Subset


filename = 'best3.pt'
traindata, validdata, testdata = get_dataloader(
    '/home/bagus/github/multibench/avmnist', num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=40, shuffle=True, num_workers=0)
validdata = DataLoader(Subset(validdata.dataset, range(500)), batch_size=40, shuffle=False, num_workers=0)
channels = 6
encoders = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
mult_head = MLP(channels*40, 100, 10).to(device)
uni_head = [MLP(channels*8, 100, 10).to(device), MLP(channels*32, 100, 10).to(device)]

fusion = Concat().to(device)

train(encoders, mult_head, uni_head, fusion, traindata, validdata, 2,
      gb_epoch=1, optimtype=torch.optim.SGD, lr=0.01, savedir=filename)

print("Testing:")
model = torch.load(filename, weights_only=False).to(device)
test(model, testdata, dataset='avmnist', no_robust=True)
