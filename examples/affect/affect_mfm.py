import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.MVAE import TSEncoder, TSDecoder # noqa
from utils.helper_modules import Sequential2 # noqa
from objective_functions.objectives_for_supervised_learning import MFM_objective # noqa
from torch import nn # noqa
from unimodals.common_models import MLP # noqa
from training_structures.Supervised_Learning import train, test # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa


classes = 2
n_latent = 256
dim_0 = 35
dim_1 = 74
dim_2 = 300
timestep = 50

# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = get_dataloader(
    'data/affect/mosi_raw.pkl', task='classification', robust_test=False, max_pad=True, max_seq_len=timestep, num_workers=0)

encoders = [TSEncoder(dim_0, 30, n_latent, timestep, returnvar=False).to(device), TSEncoder(
    dim_1, 30, n_latent, timestep, returnvar=False).to(device), TSEncoder(dim_2, 30, n_latent, timestep, returnvar=False).to(device)]

decoders = [TSDecoder(dim_0, 30, n_latent, timestep).to(device), TSDecoder(
    dim_1, 30, n_latent, timestep).to(device), TSDecoder(dim_2, 30, n_latent, timestep).to(device)]

fuse = Sequential2(Concat(), MLP(3*n_latent, n_latent, n_latent//2)).to(device)

intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(device), MLP(n_latent,
                                                                     n_latent//2, n_latent//2).to(device), MLP(n_latent, n_latent//2, n_latent//2).to(device)]

head = MLP(n_latent//2, 20, classes).to(device)

argsdict = {'decoders': decoders, 'intermediates': intermediates}

additional_modules = decoders+intermediates

objective = MFM_objective(2.0, [torch.nn.MSELoss(
), torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0, 1.0])

train(encoders, fuse, head, traindata, validdata, 2, additional_modules,
      objective=objective, objective_args_dict=argsdict, save='results/models/mosi_mfm_best.pt')

print("Testing:")
model = torch.load('results/models/mosi_mfm_best.pt', weights_only=False).to(device)

test(model=model, test_dataloaders_all=test_robust,
     dataset='mosi', is_packed=False, no_robust=True)
