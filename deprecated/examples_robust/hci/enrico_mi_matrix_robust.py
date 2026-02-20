from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from robustness.all_in_one import general_train, general_test
from datasets.enrico.get_data_robust import get_dataloader_robust
from datasets.enrico.get_data import get_dataloader
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
from torch import nn
sys.path.append(os.getcwd())


dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, _ = dls
robustdata = get_dataloader_robust('datasets/enrico/dataset')
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(device)
# encoders=[VGG16Slim(64).to(device), DAN(4, 16, dropout=True, dropoutp=0.25).to(device), DAN(28, 16, dropout=True, dropoutp=0.25).to(device)]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).to(device), DAN(28, 16, dropout=True, dropoutp=0.25).to(device)]
head = Linear(32, 20).to(device)

# fusion=Concat().to(device)
fusion = MultiplicativeInteractions2Modal([16, 16], 32, "matrix").to(device)

allmodules = encoders + [head, fusion]


def trainprocess(filename):
    train(encoders, fusion, head, traindata, validdata, 50,
          optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0, save=filename)


filename = general_train(trainprocess, 'enrico_mi_matrix')


def testprocess(model, testdata):
    return test(model, testdata)


general_test(testprocess, filename, robustdata)
