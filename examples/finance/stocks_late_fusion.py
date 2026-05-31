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

from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from training_structures.Supervised_Learning import train, test # noqa
from datasets.stocks.get_data import get_dataloader # noqa
from unimodals.common_models import LSTM, Identity # noqa
from fusions.common_fusions import ConcatWithLinear # noqa



parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


import datetime
stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader_dict = get_dataloader(
    stocks, stocks, [args.target_stock], cuda=False,
    start_date=datetime.datetime(2010, 1, 1), end_date=datetime.datetime(2021, 1, 1),
    window_size=50, val_split=1500, test_split=2000)
test_loader = test_loader_dict['timeseries'][0]

n_modalities = len(train_loader.dataset[0]) - 1
encoders = [LSTM(1, 16).to(device) for _ in range(n_modalities)]
fusion = ConcatWithLinear(n_modalities * 16, 1).to(device)
head = Identity().to(device)
allmodules = [*encoders, fusion, head]


def trainprocess():
    train(encoders, fusion, head, train_loader, val_loader, total_epochs=4,
          task='regression', optimtype=torch.optim.Adam, objective=nn.MSELoss(),
          save='results/models/stocks_lf_best.pt')


all_in_one_train(trainprocess, allmodules)

model = torch.load('results/models/stocks_lf_best.pt', weights_only=False).to(device)
# dataset = 'finance F&B', finance tech', finance health'
test(model, test_loader, dataset='finance F&B',
     task='regression', criterion=nn.MSELoss(), no_robust=True)
