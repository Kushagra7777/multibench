import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import Transformer, MLP, Sequential, Identity # noqa
from training_structures.Supervised_Learning import train, test # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import ConcatEarly # noqa


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, testdata = get_dataloader(
    'data/affect/mosi_raw.pkl',
    robust_test=False, max_pad=True, num_workers=0)

# mosi/mosei
encoders = [Identity().to(device), Identity().to(device), Identity().to(device)]
head = Sequential(Transformer(409, 300).to(device), MLP(300, 128, 1)).to(device)

# humor/sarcasm
# encoders = [Identity().to(device),Identity().to(device),Identity().to(device)]
# head = Sequential(Transformer(752, 300).to(device),MLP(300, 128, 1)).to(device)


fusion = ConcatEarly().to(device)

train(encoders, fusion, head, traindata, validdata, 2, task="regression", optimtype=torch.optim.AdamW, is_packed=False, early_stop=False, lr=1e-4, save='results/models/mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('results/models/mosi_ef_best.pt', weights_only=False).to(device)
test(model, testdata, 'affect', is_packed=False,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
