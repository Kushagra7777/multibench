from private_test_scripts.all_in_one import all_in_one_train
from training_structures.Supervised_Learning import train, test
from fusions.import MULTModel
from unimodals.common_models import MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import ConcatEarly
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, testdata = get_dataloader(
    os.path.expanduser('~/multibench/affect/processed/mosi_raw.pkl'))

# mosi/mosei
encoders = [MULTModel(409, 512).to(device)]
head = MLP(512, 256, 1).to(device)

# humor/sarcasm
# encoders = [Transformer(early=True).to(device)]
# head = MLP(1128, 512, 1).to(device)

all_modules = [*encoders, head]

fusion = ConcatEarly().to(device)


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 1000, task="regression", optimtype=torch.optim.AdamW,
          lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


all_in_one_train(trainprocess, all_modules)

print("Testing:")
model = torch.load('mosi_ef_best.pt', weights_only=False).to(device)
test(model, testdata, True, torch.nn.L1Loss(), "regression")
