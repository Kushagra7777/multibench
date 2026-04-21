from private_test_scripts.all_in_one import all_in_one_train
from training_structures.unimodal import train, test
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


modal_num = 2
# mosi
# encoders=GRU(20,50,dropout=True,has_padding=True).to(device)
# encoders=GRU(5,15,dropout=True,has_padding=True).to(device)
encoders = GRU(300, 600, dropout=True, has_padding=True).to(device)
# head=MLP(50,50,1).to(device)
# head = MLP(15, 15, 1).to(device)
head = MLP(600, 300, 1).to(device)

# mosei/iemocap

# encoders=GRU(35,70,dropout=True,has_padding=True).to(device)
# encoders=GRU(74,150,dropout=True,has_padding=True).to(device)
# encoders=GRU(300,600,dropout=True,has_padding=True).to(device)

# head = MLP(70, 35, 1).to(device)
# head = MLP(600, 300, 1).to(device)
print(encoders)
# head=MLP(820,400,1).to(device)

all_modules = [encoders, head]
# Support simple late_fusion and late_fusion with removing bias
# Simply change regularization=True
# mosi/mosei


def trainprocess():
    train(encoders, head, traindata, validdata, 1000, True, task="regression",
          optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01, modalnum=modal_num)


all_in_one_train(trainprocess, all_modules)


print("Testing:")
encoder = torch.load('encoder.pt', weights_only=False).to(device)
head = torch.load('head.pt', weights_only=False).to(device)
test(encoder, head, testdata, True, "regression", criterion=torch.nn.L1Loss())
