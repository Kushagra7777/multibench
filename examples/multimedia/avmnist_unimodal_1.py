import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from training_structures.unimodal import train, test
from torch.utils.data import DataLoader, Subset


modalnum = 1
traindata, validdata, testdata = get_dataloader(
    'data/avmnist', num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=40, shuffle=True, num_workers=0)
channels = 6
# encoders=[LeNet(1,channels,3).to(device),LeNet(1,channels,5).to(device)]
encoder = LeNet(1, channels, 5).to(device)
head = MLP(channels*32, 100, 10).to(device)


train(encoder, head, traindata, validdata, 2, optimtype=torch.optim.SGD,
      lr=0.1, weight_decay=0.0001, modalnum=modalnum,
      save_encoder='avmnist_u1_encoder.pt', save_head='avmnist_u1_head.pt')

print("Testing:")
encoder = torch.load('results/models/avmnist_u1_encoder.pt', weights_only=False).to(device)
head = torch.load('results/models/avmnist_u1_head.pt', weights_only=False)
test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
