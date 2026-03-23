import sys
import os
from torch import nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
from utils.helper_modules import Sequential2
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test
from datasets.avmnist.get_data import get_dataloader
from objective_functions.objectives_for_supervised_learning import CCA_objective
from torch.utils.data import DataLoader, Subset


traindata, validdata, testdata = get_dataloader(
    '/home/bagus/github/multibench/avmnist', batch_size=800, num_workers=0)
traindata = DataLoader(Subset(traindata.dataset, range(2000)), batch_size=800, shuffle=True, num_workers=0)
channels = 6
encoders = [LeNet(1, channels, 3).to(device), Sequential2(
    LeNet(1, channels, 5), Linear(192, 48, xavier_init=True)).to(device)]
#encoders=[MLP(300,512,outdim), MLP(4096,1024,outdim)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
#encoders=[Linear(300, 512), Linear(4096,512)]
# head=MLP(2*outdim,2*outdim,23).to(device)
head = Linear(96, 10, xavier_init=True).to(device)
fusion = Concat().to(device)

train(encoders, fusion, head, traindata, validdata, 2,
      save="best_cca.pt", optimtype=torch.optim.AdamW, lr=1e-2, objective=CCA_objective(48), objective_args_dict={})
# ,weight_decay=0.01)

print("Testing:")
model = torch.load('best_cca.pt', weights_only=False).to(device)
test(model, testdata, no_robust=True)
