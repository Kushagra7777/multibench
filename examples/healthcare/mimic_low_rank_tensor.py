import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from fusions.common_fusions import LowRankTensorFusion # noqa
from training_structures.Simple_Late_Fusion import train, test # noqa


# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).to(device), GRU(
    12, 30, dropout=False, batch_first=True).to(device)]
head = MLP(100, 40, 2, dropout=False).to(device)
fusion = LowRankTensorFusion([10, 720], 100, 40).to(device)

# train
train(encoders, fusion, head, traindata, validdata, 50, auprc=True)

# test
print("Testing: ")
model = torch.load('results/models/best.pt', weights_only=False).to(device)
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 1', auprc=True)
