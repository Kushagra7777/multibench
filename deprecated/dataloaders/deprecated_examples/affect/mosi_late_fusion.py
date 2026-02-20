from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.getcwd())


# Support mosi/mosi_unaligned/mosei/mosei_unaligned/iemocap/iemocap_unaligned
traindata, validdata, testdata = get_dataloader(
    '../affect/processed/mosi_data.pkl')

# mosi
encoders = [GRU(20, 50, dropout=True, has_padding=True).to(device),
            GRU(5, 15, dropout=True, has_padding=True).to(device),
            GRU(300, 600, dropout=True, has_padding=True).to(device)]
head = MLP(665, 300, 1).to(device)
# mosei/iemocap
'''
encoders=[GRU(35,70,dropout=True,has_padding=True).to(device), \
    GRU(74,150,dropout=True,has_padding=True).to(device),\
    GRU(300,600,dropout=True,has_padding=True).to(device)]
head=MLP(820,400,1).to(device)
'''
# iemocap
'''
encoders=[GRU(35,70,dropout=True,has_padding=True).to(device), \
    GRU(74,150,dropout=True,has_padding=True).to(device),\
    GRU(300,600,dropout=True,has_padding=True).to(device)]
head=MLP(820,400,4).to(device)
'''
fusion = Concat().to(device)

# Support simple late_fusion and late_fusion with removing bias
# Simply change regularization=True
# mosi/mosei
train(encoders, fusion, head, traindata, validdata, 1000, True, True,
      task="regression", optimtype=torch.optim.AdamW, lr=1e-4, save='best.pt',
      weight_decay=0.01, criterion=torch.nn.L1Loss(), regularization=False)
# iemocap
'''
train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    optimtype=torch.optim.AdamW,lr=1e-4,save='best.pt', \
    weight_decay=0.01,regularization=False)
'''

print("Testing:")
model = torch.load('best.pt', weights_only=False).to(device)
test(model, testdata, True, torch.nn.L1Loss(), "regression",)
# test(model,testdata,True,)
