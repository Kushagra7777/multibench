import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test
from torch.utils.data import DataLoader, Subset


traindata, validdata, testdata = get_dataloader(
    'data/avmnist', num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=40, shuffle=True, num_workers=0)
channels = 6
encoders = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
head = MLP(channels*40, 100, 10).to(device)

fusion = Concat().to(device)

train(encoders, fusion, head, traindata, validdata, 2,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001,
      save='results/models/avmnist_slf_best.pt')

print("Testing:")
model = torch.load('results/models/avmnist_slf_best.pt', weights_only=False).to(device)
test(model, testdata, no_robust=True)
