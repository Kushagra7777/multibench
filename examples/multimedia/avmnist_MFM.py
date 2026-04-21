import sys
import os
sys.path.append(os.getcwd())

from objective_functions.recon import recon_weighted_sum, sigmloss1dcentercrop
from unimodals.MVAE import LeNetEncoder, DeLeNet
from training_structures.Supervised_Learning import train, test
from objective_functions.objectives_for_supervised_learning import MFM_objective
from utils.helper_modules import Sequential2
from datasets.avmnist.get_data import get_dataloader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from unimodals.common_models import MLP
from fusions.common_fusions import Concat
from torch.utils.data import DataLoader, Subset


traindata, validdata, testdata = get_dataloader(
    'data/avmnist', num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=40, shuffle=True, num_workers=0)
channels = 6

classes = 10
n_latent = 200
fuse = Concat()
fuse = Sequential2(Concat(), MLP(2*n_latent, n_latent, n_latent//2)).to(device)
encoders = [LeNetEncoder(1, channels, 3, n_latent, twooutput=False).to(device), LeNetEncoder(1, channels, 5, n_latent, twooutput=False).to(device)]
decoders = [DeLeNet(1, channels, 3, n_latent).to(device),
            DeLeNet(1, channels, 5, n_latent).to(device)]

intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(device),
                 MLP(n_latent, n_latent//2, n_latent//2).to(device)]
head = MLP(n_latent//2, 40, classes).to(device)
objective = MFM_objective(2.0, [sigmloss1dcentercrop(
    28, 34), sigmloss1dcentercrop(112, 130)], [1.0, 1.0])
train(encoders, fuse, head, traindata, validdata, 2, decoders+intermediates,
      objective=objective, objective_args_dict={'decoders': decoders, 'intermediates': intermediates},
      save='avmnist_mfm_best.pt')
model = torch.load('avmnist_mfm_best.pt', weights_only=False)
test(model, testdata, no_robust=True)
