from unimodals.common_models import LeNet, MLP, Constant
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.getcwd())

filename = 'bestmi.pt'
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
head = MLP(channels*40, 100, 10).to(device)

# fusion=Concat().to(device)
fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*40, 'matrix')

train(encoders, fusion, head, traindata, validdata, 20,
      optimtype=torch.optim.SGD, lr=0.05, weight_decay=0.0001, save=filename)

print("Testing:")
model = torch.load(filename, weights_only=False).to(device)
test(model, testdata)
