"""
MultiBench Example Usage - Part 3: MOSI (L2-MCTN)

Demonstrates MultiBench's L2-MCTN (Multimodal Cyclic Translation Network, Level 2)
training paradigm on the MOSI sentiment analysis dataset.
"""

import sys
import os

# Add the MultiBench root directory to the path so imports resolve correctly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

# Verify data file
data_path = os.path.join(repo_root, 'data', 'affect', 'mosi_raw.pkl')
print(f"Data path: {data_path}")
print(f"Data file exists: {os.path.exists(data_path)}")

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from datasets.affect.get_data import get_dataloader  # noqa

traindata, validdata, testdata = \
    get_dataloader(data_path, robust_test=False)

from unimodals.common_models import GRU, MLP  # noqa
from fusions.MCTN import Encoder, Decoder  # noqa

max_seq = 20
feature_dim = 300
hidden_dim = 32

encoder0 = Encoder(feature_dim, hidden_dim, n_layers=1, dropout=0.0).to(device)
decoder0 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(device)
encoder1 = Encoder(hidden_dim, hidden_dim, n_layers=1, dropout=0.0).to(device)
decoder1 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(device)

reg_encoder = nn.GRU(hidden_dim, 32).to(device)

from unimodals.common_models import MLP  # noqa
head = MLP(32, 64, 1).to(device)

from private_test_scripts.all_in_one import all_in_one_train  # noqa
from training_structures.MCTN_Level2 import train, test  # noqa

allmodules = [encoder0, decoder0, encoder1, decoder1, reg_encoder, head]


def trainprocess():
    train(
        traindata, validdata,
        encoder0, decoder0, encoder1, decoder1,
        reg_encoder, head,
        criterion_t0=nn.MSELoss(), criterion_c=nn.MSELoss(),
        criterion_t1=nn.MSELoss(), criterion_r=nn.L1Loss(),
        max_seq_len=20,
        mu_t0=0.01, mu_c=0.01, mu_t1=0.01,
        dropout_p=0.15, early_stop=False, patience_num=15,
        lr=1e-4, weight_decay=0.01, op_type=torch.optim.AdamW,
        epoch=200, model_save='best_mctn.pt')


all_in_one_train(trainprocess, allmodules)

model = torch.load('best_mctn.pt', map_location=device, weights_only=False).to(device)

test(model, testdata, 'mosi', no_robust=True)
