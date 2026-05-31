from torch import nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os

sys.path.append(os.getcwd()) 
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))



from training_structures.gradient_blend import train, test # noqa
from unimodals.common_models import GRU, MLP, Transformer # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = \
    get_dataloader('data/affect/mosi_raw.pkl',
                   task='classification', robust_test=False, max_pad=True, num_workers=0)

# mosi/mosei
encoders = [Transformer(35, 70).to(device),
            Transformer(74, 150).to(device),
            Transformer(300, 600).to(device)]
head = MLP(820, 512, 2).to(device)

unimodal_heads = [MLP(70, 32, 2).to(device), MLP(
    150, 64, 2).to(device), MLP(600, 256, 2).to(device)]

# humor/sarcasm
# encoders=[Transformer(371,700).to(device), \
#     Transformer(81,150).to(device),\
#     Transformer(300,600).to(device)]
# head=MLP(1450,512,2).to(device)

# unimodal_heads=[MLP(700,512,2).to(device),MLP(150,64,2).to(device),MLP(600,256,2).to(device)]

fusion = Concat().to(device)

# training_structures.gradient_blend.criterion = nn.L1Loss()

train(encoders, head, unimodal_heads, fusion, traindata, validdata, 2, gb_epoch=1, lr=1e-3, AUPRC=False,
      classification=True, optimtype=torch.optim.AdamW, savedir='results/models/mosi_best_gb.pt', weight_decay=0.1)

print("Testing:")
model = torch.load('results/models/mosi_best_gb.pt', weights_only=False).to(device)

test(model, test_robust, dataset='mosi', auprc=False, no_robust=True)

# test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification')
