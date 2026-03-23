import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
from training_structures.Supervised_Learning import train, test
from torch.utils.data import DataLoader, Subset


filename = 'bestmi.pt'
traindata, validdata, testdata = get_dataloader(
    '/home/bagus/github/multibench/avmnist', num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=40, shuffle=True, num_workers=0)
channels = 6
encoders = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
head = MLP(channels*40, 100, 10).to(device)

# fusion=Concat().to(device)
fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*40, 'matrix')

train(encoders, fusion, head, traindata, validdata, 2,
      optimtype=torch.optim.SGD, lr=0.05, weight_decay=0.0001, save=filename)

print("Testing:")
model = torch.load(filename, weights_only=False).to(device)
test(model, testdata, no_robust=True)
