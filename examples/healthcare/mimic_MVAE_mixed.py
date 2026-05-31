import sys
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
sys.path.append(os.getcwd())

from unimodals.MVAE import MLPEncoder, TSEncoder, TSDecoder # noqa
from objective_functions.recon import elbo_loss, sigmloss1d # noqa
from training_structures.Supervised_Learning import train, test # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from objective_functions.objectives_for_supervised_learning import MVAE_objective # noqa
from unimodals.common_models import MLP # noqa
from fusions.MVAE import ProductOfExperts_Zipped # noqa


traindata, validdata, testdata = get_dataloader(
    7, imputed_path=os.path.expanduser('~/yiwei/im.pk'), flatten_time_series=True)
classes = 2
n_latent = 200
series_dim = 12
timestep = 24
fuse = ProductOfExperts_Zipped((1, 40, n_latent))
encoders = [MLPEncoder(5, 20, n_latent).to(device), TSEncoder(
    series_dim, 30, n_latent, timestep, batch_first=True).to(device)]
decoders = [MLP(n_latent, 20, 5).to(device), TSDecoder(
    series_dim, 30, n_latent, timestep).to(device)]
head = MLP(n_latent, 20, classes).to(device)
elbo = MVAE_objective(2.0, [sigmloss1d, sigmloss1d], [1.0, 1.0], annealing=0.0)
argsdict = {'decoders': decoders}
train(encoders, fuse, head, traindata, validdata, 30, decoders,
      optimtype=torch.optim.Adam, lr=0.0001, objective=elbo, objective_args_dict=argsdict)

model = torch.load('results/models/best.pt', weights_only=False)
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 7')
