import sys
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
sys.path.append(os.getcwd())


from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned # noqa
from memory_profiler import memory_usage # noqa
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from datasets.enrico.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal # noqa
from training_structures.Supervised_Learning import train, test # noqa


dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(device)
# encoders=[VGG16Slim(64).to(device), DAN(4, 16, dropout=True, dropoutp=0.25).to(device), DAN(28, 16, dropout=True, dropoutp=0.25).to(device)]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).to(device), DAN(28, 16, dropout=True, dropoutp=0.25).to(device)]
head = Linear(32, 20).to(device)

# fusion=Concat().to(device)
fusion = MultiplicativeInteractions2Modal([16, 16], 32, "matrix", True).to(device)

allmodules = encoders + [head, fusion]


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 50,
          optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0)


all_in_one_train(trainprocess, allmodules)

print("Testing:")
model = torch.load('best.pt', weights_only=False).to(device)

test(model, testdata, dataset='enrico')
