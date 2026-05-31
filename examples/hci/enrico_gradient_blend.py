import sys
import os
from torch import nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append(os.getcwd())

from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned # noqa
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from datasets.enrico.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.gradient_blend import train, test # noqa


dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(device)
# encoders=[VGG16Slim(64).to(device), DAN(4, 16, dropout=True, dropoutp=0.25).to(device), DAN(28, 16, dropout=True, dropoutp=0.25).to(device)]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).to(device), DAN(28, 16, dropout=True, dropoutp=0.25).to(device)]
mult_head = Linear(32, 20).to(device)
uni_head = [Linear(16, 20).to(device), Linear(16, 20).to(device)]

fusion = Concat().to(device)

# train(encoders,fusion,head,traindata,validdata,num_epoch=50,gb_epoch=10,optimtype=torch.optim.Adam,lr=0.0001,weight_decay=0)
allmodules = encoders + [mult_head, fusion] + uni_head


def trainprocess():
    train(encoders, mult_head, uni_head, fusion, traindata, validdata, 2,
          gb_epoch=1, optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0,
          savedir='results/models/enrico_best.pt')


all_in_one_train(trainprocess, allmodules)


model = torch.load('results/models/enrico_best.pt', weights_only=False).to(device)
clean_testdata = testdata[list(testdata.keys())[0]][0]
test(model, clean_testdata, dataset='enrico', no_robust=True)
