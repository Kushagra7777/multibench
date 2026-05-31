from torch import nn
import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pmdarima
import numpy as np
import argparse
import sys
import os

sys.path.append(os.getcwd())

from training_structures.unimodal import train, test # noqa
from datasets.stocks.get_data import get_dataloader, Grouping # noqa
from fusions.mult import MULTModel # noqa
from unimodals.common_models import Identity # noqa


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


import datetime
stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader_dict = get_dataloader(
    stocks, stocks, [args.target_stock], modality_first=False, cuda=False,
    start_date=datetime.datetime(2010, 1, 1), end_date=datetime.datetime(2021, 1, 1),
    window_size=50, val_split=1500, test_split=2000)
test_loader = test_loader_dict['timeseries'][0]

n_modalities = 3
grouping = Grouping(n_modalities)

# Get n_features for each group
n_features = [x.size(-1) for x in grouping(next(iter(train_loader))[0])]

model = nn.Sequential(grouping, MULTModel(n_modalities, n_features)).to(device)
identity = Identity()
allmodules = [model, identity]


def trainprocess():
    train(model, identity, train_loader, val_loader,
          total_epochs=4, task='regression',
          optimtype=torch.optim.Adam, criterion=nn.MSELoss())


trainprocess()

encoder = torch.load('results/models/encoder.pt', weights_only=False).to(device)
head = torch.load('results/models/head.pt', weights_only=False).to(device)
# dataset = 'finance F&B', finance tech', finance health'
test(encoder, head, test_loader, dataset='finance F&B',
     task='regression', criterion=nn.MSELoss(), no_robust=True)
