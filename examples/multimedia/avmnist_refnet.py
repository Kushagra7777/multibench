import sys
import os
sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
from objective_functions.objectives_for_supervised_learning import RefNet_objective
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.helper_modules import Sequential2
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test
from torch.utils.data import DataLoader, Subset

traindata, validdata, testdata = get_dataloader(
    '/home/bagus/github/multibench/avmnist', batch_size=20, num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=20, shuffle=True, num_workers=0)
channels = 6
encoders = [Sequential2(LeNet(1, channels, 3), nn.Linear(
    channels*8, channels*32)).to(device), LeNet(1, channels, 5).to(device)]
head = MLP(channels*64, 100, 10).to(device)
refiner = MLP(channels*64, 1000, 13328).to(device)
fusion = Concat().to(device)

train(encoders, fusion, head, traindata, validdata, 2, [
      refiner], optimtype=torch.optim.SGD, lr=0.005, objective=RefNet_objective(0.1), objective_args_dict={'refiner': refiner},
      save='avmnist_refnet_best.pt')

print("Testing:")
model = torch.load('avmnist_refnet_best.pt', weights_only=False).to(device)
test(model, testdata, no_robust=True)
