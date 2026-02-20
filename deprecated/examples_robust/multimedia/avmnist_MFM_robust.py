from objective_functions.recon import recon_weighted_sum, sigmloss1dcentercrop
from unimodals.MVAE import LeNetEncoder, DeLeNet
from training_structures.MFM import train_MFM, test_MFM
from datasets.avmnist.get_data_robust import get_dataloader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from unimodals.common_models import MLP
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


filename = 'avmnist_MFM_robust_best.pt'
traindata, validdata, testdata, robustdata = get_dataloader(
    '../../../../yiwei/avmnist/_MFAS/avmnist')
channels = 6

classes = 10
n_latent = 200
fuse = Concat()

encoders = [LeNetEncoder(1, channels, 3, n_latent, twooutput=False).to(device), LeNetEncoder(1, channels, 5, n_latent, twooutput=False).to(device)]
decoders = [DeLeNet(1, channels, 3, n_latent).to(device),
            DeLeNet(1, channels, 5, n_latent).to(device)]

intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(device), MLP(n_latent,
                                                                     n_latent//2, n_latent//2).to(device), MLP(2*n_latent, n_latent, n_latent//2).to(device)]
head = MLP(n_latent//2, 40, classes).to(device)
recon_loss = recon_weighted_sum([sigmloss1dcentercrop(
    28, 34), sigmloss1dcentercrop(112, 130)], [1.0, 1.0])

train_MFM(encoders, decoders, head, intermediates, fuse,
          recon_loss, traindata, validdata, 25, savedir=filename)

model = torch.load(filename, weights_only=False)
print("Testing:")
test_MFM(model, testdata)

print("Robustness testing:")
test(model, testdata)
