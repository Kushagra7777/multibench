import sys
import os

sys.path.append(os.getcwd())

from objective_functions.recon import elbo_loss, sigmloss1dcentercrop
from unimodals.MVAE import LeNetEncoder, DeLeNet
from training_structures.Supervised_Learning import train, test
from datasets.avmnist.get_data import get_dataloader
from objective_functions.objectives_for_supervised_learning import MVAE_objective
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from unimodals.common_models import MLP
from fusions.MVAE import ProductOfExperts_Zipped
from torch.utils.data import DataLoader, Subset


traindata, validdata, testdata = get_dataloader(
    'data/avmnist', num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=40, shuffle=True, num_workers=0)

classes = 10
n_latent = 200
fuse = ProductOfExperts_Zipped((1, 40, n_latent))


channels = 6
encoders = [LeNetEncoder(1, channels, 3, n_latent).to(device), LeNetEncoder(1, channels, 5, n_latent).to(device)]
decoders = [DeLeNet(1, channels, 3, n_latent).to(device),
            DeLeNet(1, channels, 5, n_latent).to(device)]
head = MLP(n_latent, 40, classes).to(device)
elbo = MVAE_objective(2.0, [sigmloss1dcentercrop(
    28, 34), sigmloss1dcentercrop(112, 130)], [1.0, 1.0], annealing=0.0)
train(encoders, fuse, head, traindata, validdata, 2, decoders,
      objective=elbo, objective_args_dict={'decoders': decoders},
      save='results/models/avmnist_mvae_best.pt')
mvae = torch.load('results/models/avmnist_mvae_best.pt', weights_only=False)
test(mvae, testdata, no_robust=True)
