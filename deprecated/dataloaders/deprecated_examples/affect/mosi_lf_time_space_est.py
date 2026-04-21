from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from unimodals.common_models import GRU, MLP
from get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # __file__ father path
print(BASE_DIR)
sys.path.append(BASE_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# analysis the time and space complexity


# Support mosi/mosi_unaligned/mosei/mosei_unaligned/iemocap/iemocap_unaligned
traindata, validdata, testdata = get_dataloader(
    os.path.expanduser('~/multibench/affect/processed/mosei_senti_data.pkl'))

# mosi
# encoders=[GRU(20,50,dropout=True,has_padding=True).to(device), \
#     GRU(5,15,dropout=True,has_padding=True).to(device),\
#     GRU(300,600,dropout=True,has_padding=True).to(device)]
# head=MLP(665,300,1).to(device)

# mosei/iemocap

encoders = [GRU(35, 70, dropout=True, has_padding=True).to(device),
            GRU(74, 150, dropout=True, has_padding=True).to(device),
            GRU(300, 600, dropout=True, has_padding=True).to(device)]
head = MLP(820, 400, 1).to(device)

# iemocap
'''
encoders=[GRU(35,70,dropout=True,has_padding=True).to(device), \
    GRU(74,150,dropout=True,has_padding=True).to(device),\
    GRU(300,600,dropout=True,has_padding=True).to(device)]
head=MLP(820,400,4).to(device)
'''
fusion = Concat().to(device)
allmodules = [encoders[0], encoders[1], encoders[2], head, fusion]
# Support simple late_fusion and late_fusion with removing bias
# Simply change regularization=True
# mosi/mosei


def trainprocess():
    # train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    #     task="regression",optimtype=torch.optim.AdamW,lr=1e-4,save='best.pt', \
    #     weight_decay=0.01,criterion=torch.nn.L1Loss(),regularization=False)
    train(encoders, fusion, head, traindata, validdata, 1000, True, True,
          task="regression", optimtype=torch.optim.AdamW, lr=1e-4, save='best.pt',
          weight_decay=0.01, criterion=torch.nn.MSELoss(), regularization=False)


all_in_one_train(trainprocess, allmodules)
# iemocap
'''
train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    optimtype=torch.optim.AdamW,lr=1e-4,save='best.pt', \
    weight_decay=0.01,regularization=False)
'''

print("Testing:")
model = torch.load('best.pt', weights_only=False).to(device)


def testprocess():
    test(model, testdata, True, torch.nn.L1Loss(), "regression")


all_in_one_test(testprocess, [model])
# test(model,testdata,True,)
