from torch import nn
import torch.nn.functional as F
import torch
import pmdarima
import numpy as np
import argparse
import sys
import os
sys.path.append(os.getcwd())

from datasets.stocks.get_data import get_dataloader # noqa
from unimodals.common_models import LSTM # noqa
from fusions.common_fusions import Stack # noqa



parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


import datetime
stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader_dict = get_dataloader(
    stocks, stocks, [args.target_stock], modality_first=True, cuda=False,
    start_date=datetime.datetime(2010, 1, 1), end_date=datetime.datetime(2021, 1, 1),
    window_size=50, val_split=1500, test_split=2000)
test_loader = test_loader_dict['timeseries'][0]


def baselines():
    def best_constant(y_prev, y):
        return float(nn.MSELoss()(torch.ones_like(y) * torch.mean(y), y))

    def copy_last(y_prev, y):
        return nn.MSELoss()(torch.cat([y_prev[-1:], y[:-1]]), y).item()

    def arima(y_prev, y):
        arr = y_prev.cpu().numpy().flatten()
        arima_model = pmdarima.arima.auto_arima(arr)
        pred = arima_model.predict(len(y))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return nn.MSELoss()(torch.tensor(pred, device=device, dtype=torch.float32).reshape(y.shape), y.to(device))

    print('Best constant val MSE loss: ' +
          str(best_constant(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('Best constant test MSE loss: ' +
          str(best_constant(val_loader.dataset.Y, test_loader.dataset.Y)))
    print('Copy-last val MSE loss: ' +
          str(copy_last(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('Copy-last test MSE loss: ' +
          str(copy_last(val_loader.dataset.Y, test_loader.dataset.Y)))
    print('ARIMA val MSE loss: ' +
          str(arima(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('ARIMA test MSE loss: ' +
          str(arima(torch.cat([train_loader.dataset.Y, val_loader.dataset.Y]), test_loader.dataset.Y)))


baselines()
