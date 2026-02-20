from unimodals.common_models import LeNet, MLP, Constant
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from datasets.avmnist.get_data_robust import get_dataloader
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

traindata, validdata, testdata, robustdata = get_dataloader(
    '../../../../yiwei/avmnist/_MFAS/avmnist')
channels = 3
encoders = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
head = MLP(channels*32, 100, 10).to(device)

fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*32, 'matrix', True).to(device)
# fusion=MultiplicativeInteractions2Modal([channels*32,channels*8],channels*32,'vector',True,flip=True).to(device)

train(encoders, fusion, head, traindata, validdata, 100, optimtype=torch.optim.SGD,
      lr=0.01, weight_decay=0.0001, save='avmnist_tensor_matrix_robust_best.pt')

model = torch.load('avmnist_tensor_matrix_robust_best.pt', weights_only=False).to(device)
print("Testing:")
test(model, testdata)

print("Robustness testing:")
test(model, testdata)
