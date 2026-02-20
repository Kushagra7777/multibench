import sys
import os
sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from training_structures.unimodal import train, test

modalnum = 0
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 3
# encoders=[LeNet(1,channels,3).to(device),LeNet(1,channels,5).to(device)]
encoder = LeNet(1, channels, 3).to(device)
head = MLP(channels*8, 100, 10).to(device)


train(encoder, head, traindata, validdata, 20, optimtype=torch.optim.SGD,
      lr=0.01, weight_decay=0.0001, modalnum=modalnum)

print("Testing:")
encoder = torch.load('encoder.pt', weights_only=False).to(device)
head = torch.load('head.pt', weights_only=False)
test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
