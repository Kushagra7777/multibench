from unimodals.common_models import LeNet, MLP, Constant, Linear
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 3
encoders = LeNet(1, channels, 5).to(device)
head = Linear(channels*32, 10).to(device)
mn = 1

train(encoders, head, traindata, validdata, 100,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001, modalnum=mn)

print("Testing:")
encoder = torch.load('encoder.pt', weights_only=False).to(device)
head = torch.load('head.pt', weights_only=False)
test(encoder, head, testdata, modalnum=mn)
