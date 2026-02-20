from objective_functions.recon import recon_weighted_sum, sigmloss1d
from unimodals.MVAE import TSEncoder, TSDecoder
from training_structures.MFM import train_MFM, test_MFM
from datasets.mimic.get_data import get_dataloader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from unimodals.common_models import MLP
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    7, imputed_path='datasets/mimic/im.pk', flatten_time_series=True)
classes = 2
n_latent = 200
series_dim = 12
timestep = 24
fuse = Concat()
encoders = [MLP(5, 20, n_latent).to(device), TSEncoder(
    series_dim, 30, n_latent, timestep, returnvar=False).to(device)]
decoders = [MLP(n_latent, 20, 5).to(device), TSDecoder(
    series_dim, 30, n_latent, timestep).to(device)]
intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(device), MLP(n_latent,
                                                                     n_latent//2, n_latent//2).to(device), MLP(2*n_latent, n_latent, n_latent//2).to(device)]
head = MLP(n_latent//2, 20, classes).to(device)
recon_loss = recon_weighted_sum([sigmloss1d, sigmloss1d], [1.0, 1.0])
# train_MFM(encoders,decoders,head,intermediates,fuse,recon_loss,traindata,validdata,25,savedir='bestmfm.pt')
mvae = torch.load('bestmfm.pt', weights_only=False)
test_MFM(mvae, testdata)
