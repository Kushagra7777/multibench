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
    '/home/bagus/github/multibench/avmnist', batch_size=32)
import glob
best_files = sorted(glob.glob('tests/best*.pt'))
if not best_files:
    raise FileNotFoundError("No model files found in tests/best*.pt")
best_file = best_files[-1]
print(f"Loading best model: {best_file}")
model = torch.load(best_file, weights_only=False).to(device)
test(model, testdata, dataset='avmnist', no_robust=True)
