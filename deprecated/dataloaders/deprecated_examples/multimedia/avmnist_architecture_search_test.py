from unimodals.common_models import LeNet, MLP, Constant
from training_structures.architecture_search import train, test
import utils.surrogate as surr
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist', batch_size=32)
model = torch.load('temp/best.pt', weights_only=False).to(device)
test(model, testdata)
