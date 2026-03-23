import sys
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn

sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
import utils.surrogate as surr
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.architecture_search import train
from torch.utils.data import DataLoader, Subset


traindata, validdata, testdata = get_dataloader(
    '/home/bagus/github/multibench/avmnist', batch_size=32)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=32, shuffle=True)
validdata = DataLoader(Subset(validdata.dataset, range(500)), batch_size=32, shuffle=False)

s_data = train(['pretrained/avmnist/image_encoder.pt', 'pretrained/avmnist/audio_encoder.pt'], 16, 10, [(6, 12, 24), (6, 12, 24, 48, 96)],
               traindata, validdata, surr.SimpleRecurrentSurrogate().to(device), (3, 5, 2), epochs=2,
               search_iter=1, num_samples=3, epoch_surrogate=5)

"""
print("Testing:")
model=torch.load('best.pt', weights_only=False).to(device)
test(model,testdata)
"""
