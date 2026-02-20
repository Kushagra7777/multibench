import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os

sys.path.append(os.getcwd())

from utils.helper_modules import Sequential2
from unimodals.common_models import Linear, MLP, MaxOut_MLP
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from objective_functions.objectives_for_supervised_learning import MFM_objective
from objective_functions.recon import sigmloss1d
from training_structures.Supervised_Learning import train, test

filename = "best_mfm.pt"
traindata, validdata, testdata = get_dataloader(
    "../video/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=128)

classes = 23
n_latent = 512
fuse = Sequential2(Concat(), MLP(2*n_latent, n_latent, n_latent//2)).to(device)
encoders = [MaxOut_MLP(512, 512, 300, n_latent, False).to(device), MaxOut_MLP(512, 1024, 4096, n_latent, False).to(device)]
head = Linear(n_latent//2, classes).to(device)

decoders = [MLP(n_latent, 600, 300).to(device), MLP(n_latent, 2048, 4096).to(device)]
intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(device),
                 MLP(n_latent, n_latent//2, n_latent//2).to(device)]

recon_loss = MFM_objective(2.0, [sigmloss1d, sigmloss1d], [
                           1.0, 1.0], criterion=torch.nn.BCEWithLogitsLoss())

train(encoders, fuse, head, traindata, validdata, 1000, decoders+intermediates, early_stop=True, task="multilabel",
      objective_args_dict={"decoders": decoders, "intermediates": intermediates}, save=filename, optimtype=torch.optim.AdamW, lr=5e-3, weight_decay=0.01, objective=recon_loss)

print("Testing:")
model = torch.load(filename, weights_only=False).to(device)
test(model, testdata, method_name="MFM", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
