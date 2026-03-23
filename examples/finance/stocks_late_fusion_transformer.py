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

from training_structures.Supervised_Learning import train, test # noqa
from datasets.stocks.get_data import get_dataloader # noqa
from fusions.common_fusions import ConcatWithLinear, LateFusionTransformer # noqa
from unimodals.common_models import Identity # noqa


class UnsqueezeLFT(nn.Module):
    """Wrapper for LateFusionTransformer that handles 2D (batch, seq) input."""
    def __init__(self, embed_dim=9):
        super().__init__()
        self.lft = LateFusionTransformer(embed_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        return self.lft(x)



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
encoders = [UnsqueezeLFT(embed_dim=9).to(device)
            for _ in range(n_modalities)]
fusion = ConcatWithLinear(n_modalities * 9, 1).to(device)
head = Identity().to(device)
allmodules = [*encoders, fusion, head]


def trainprocess():
    train(encoders, fusion, head, train_loader, val_loader, total_epochs=4,
          task='regression', optimtype=torch.optim.Adam, objective=nn.MSELoss(),
          save='stocks_lft_best.pt')


trainprocess()

model = torch.load('stocks_lft_best.pt', weights_only=False).to(device)
# dataset = 'finance F&B', finance tech', finance health'
test(model, test_loader, dataset='finance F&B',
     task='regression', criterion=nn.MSELoss(), no_robust=True)
