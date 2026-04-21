from private_test_scripts.all_in_one import all_in_one_train
from training_structures.gradient_blend import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, testdata = get_dataloader(
    os.path.expanduser('~/MultiBench/mosi_raw.pkl'))

# mosi/mosei
encoders = [GRU(35, 70, dropout=True, has_padding=True).to(device),
            GRU(74, 150, dropout=True, has_padding=True).to(device),
            GRU(300, 600, dropout=True, has_padding=True).to(device)]
head = MLP(820, 400, 1).to(device)
unimodal_heads = [MLP(70, 50, 1).to(device), MLP(
    150, 100, 1).to(device), MLP(600, 256, 1).to(device)]

# humor/sarcasm
# encoders=[GRU(371,512,dropout=True,has_padding=True).to(device), \
#     GRU(81,256,dropout=True,has_padding=True).to(device),\
#     GRU(300,600,dropout=True,has_padding=True).to(device)]
# head=MLP(1368,512,1).to(device)

all_modules = [*encoders, head, *unimodal_heads]

fusion = Concat().to(device)


def trainprocess():
    train(encoders, head, unimodal_heads, fusion, traindata,
          validdata, 300, lr=0.005, AUPRC=False, savedir='gb.pt')


all_in_one_train(trainprocess, all_modules)

print("Testing:")
model = torch.load('gb.pt', weights_only=False).to(device)
test(model, testdata)
