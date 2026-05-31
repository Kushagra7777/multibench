import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test # noqa
from unimodals.common_models import Transformer, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa

# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = \
    get_dataloader('data/affect/mosi_data.pkl',
                   robust_test=False, max_pad=True, num_workers=0)

# mosi/mosei
encoders = [Transformer(20, 40).to(device),
            Transformer(5, 10).to(device),
            Transformer(300, 600).to(device)]
head = MLP(650, 256, 1).to(device)

# humor/sarcasm
# encoders=[Transformer(371,400).to(device), \
#     Transformer(81,100).to(device),\
#     Transformer(300,600).to(device)]
# head=MLP(1100,256,1).to(device)

fusion = Concat().to(device)

train(encoders, fusion, head, traindata, validdata, 2, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=False, lr=1e-4, save='results/models/mosi_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


print("Testing:")
model = torch.load('results/models/mosi_lf_best.pt', weights_only=False).to(device)

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
